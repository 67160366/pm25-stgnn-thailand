# [TODO: NSC Disclaimer — see booklet page 44]

"""Training loop with early stopping and optional W&B logging.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.models.base import PM25ModelBase
from src.training.losses import masked_mse
from src.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Standard PyTorch training loop for PM25ModelBase subclasses.

    Features:
        - Adam optimiser with cosine LR schedule.
        - Early stopping on val RMSE at the primary horizon (default 24h).
        - Optional Weights & Biases logging (skipped gracefully if wandb absent).
        - Checkpoint saving to ``output_dir / best_model.pt``.

    Args:
        model: Any PM25ModelBase subclass.
        train_loader: PyG DataLoader for the training split.
        val_loader: PyG DataLoader for the validation split.
        lr: Initial learning rate for Adam.
        max_epochs: Maximum training epochs.
        patience: Early-stopping patience (epochs without val improvement).
        primary_horizon: Horizon (hours) used for early stopping. Must be in
            ``model.horizons``.
        device: Torch device string or object.
        output_dir: Directory for checkpoints and logs.
        wandb_project: W&B project name. Set to None to disable W&B.
        wandb_run_name: W&B run name. Auto-generated if None.
    """

    def __init__(
        self,
        model: PM25ModelBase,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        primary_horizon: int = 24,
        device: str | torch.device = "cpu",
        output_dir: Path = Path("outputs"),
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        if primary_horizon not in model.horizons:
            raise ValueError(
                f"primary_horizon={primary_horizon} not in model.horizons={model.horizons}"
            )

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.primary_horizon = primary_horizon
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=lr * 0.01)

        self._wandb: Any = None
        if wandb_project is not None:
            self._wandb = self._init_wandb(wandb_project, wandb_run_name)

        logger.info(
            "Trainer: model=%s params=%d device=%s lr=%.1e epochs=%d patience=%d",
            type(model).__name__,
            model.count_parameters(),
            self.device,
            lr,
            max_epochs,
            patience,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            History dict with keys ``"train_loss"``, ``"val_loss"``,
            ``"val_rmse_<primary>h"`` — one float per epoch.
        """
        self.model.to(self.device)

        best_val_rmse = float("inf")
        epochs_no_improve = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            f"val_rmse_{self.primary_horizon}h": [],
        }

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._eval_epoch()
            self.scheduler.step()

            primary_rmse = val_metrics.get(f"rmse_{self.primary_horizon}h", float("nan"))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history[f"val_rmse_{self.primary_horizon}h"].append(primary_rmse)

            logger.info(
                "epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  " "val_rmse_%dh=%.2f  lr=%.2e",
                epoch,
                self.max_epochs,
                train_loss,
                val_loss,
                self.primary_horizon,
                primary_rmse,
                self.optimizer.param_groups[0]["lr"],
            )

            if self._wandb is not None:
                self._wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Early stopping
            if primary_rmse < best_val_rmse:
                best_val_rmse = primary_rmse
                epochs_no_improve = 0
                self._save_checkpoint(epoch, primary_rmse)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch,
                        self.patience,
                    )
                    break

        logger.info(
            "Training complete. Best val RMSE@%dh = %.2f", self.primary_horizon, best_val_rmse
        )
        if self._wandb is not None:
            self._wandb.finish()

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)  # (B*N, H)
            target = batch["station"].y  # (B*N, H)
            mask = batch["station"].mask  # (B*N, H) bool
            loss = masked_mse(pred, target, mask)
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at batch {n_batches} — check target scale and input features."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _eval_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_pred: list[np.ndarray] = []
        all_target: list[np.ndarray] = []
        all_mask: list[np.ndarray] = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = batch["station"].y
                mask = batch["station"].mask
                loss = masked_mse(pred, target, mask)
                total_loss += loss.item()
                n_batches += 1
                all_pred.append(pred.cpu().numpy())
                all_target.append(target.cpu().numpy())
                all_mask.append(mask.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        if all_pred:
            metrics = compute_metrics(
                pred=np.concatenate(all_pred, axis=0),
                target=np.concatenate(all_target, axis=0),
                mask=np.concatenate(all_mask, axis=0),
                horizons=self.model.horizons,
            )
        else:
            metrics = {}

        return avg_loss, metrics

    def _save_checkpoint(self, epoch: int, val_rmse: float) -> None:
        path = self.output_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                f"val_rmse_{self.primary_horizon}h": val_rmse,
                "model_class": type(self.model).__name__,
            },
            path,
        )
        logger.info("Checkpoint saved to %s (epoch=%d, val_rmse=%.2f)", path, epoch, val_rmse)

    def _init_wandb(self, project: str, run_name: str | None) -> object:
        try:
            import wandb

            run = wandb.init(
                project=project,
                name=run_name,
                config={
                    "model": type(self.model).__name__,
                    "n_stations": self.model.n_stations,
                    "n_features": self.model.n_features,
                    "horizons": self.model.horizons,
                    "params": self.model.count_parameters(),
                },
            )
            logger.info("W&B run initialised: %s", run.url if run else "unknown")
            return wandb
        except ImportError:
            logger.warning("wandb not installed — W&B logging disabled.")
            return None
