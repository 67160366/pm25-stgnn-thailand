# [TODO: NSC Disclaimer — see booklet page 44]

"""Hydra entry point for training PM2.5 STGNN models.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Usage:
    uv run python scripts/03_train.py model=a3tgcn
    uv run python scripts/03_train.py model=mtgnn trainer.device=cuda
    uv run python scripts/03_train.py model=a3tgcn trainer.max_epochs=10
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from src.data.loader import PM25GraphDataset
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Instantiate datasets, model, and trainer from Hydra config, then fit.

    Args:
        cfg: Hydra DictConfig composed from configs/config.yaml and overrides.
    """
    # Resolve device: fall back to CPU if CUDA requested but unavailable
    device = cfg.trainer.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device = "cpu"

    logger.info("=== PM2.5 STGNN Training ===")
    logger.info("Model: %s", cfg.model._target_)
    logger.info("Device: %s", device)

    # --- Datasets ---
    logger.info("Loading datasets …")
    common_ds_kwargs = dict(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        window_in=cfg.data.window_in,
        horizons=list(cfg.data.horizons),
        exclude_stations=list(cfg.data.exclude_stations),
    )
    train_ds = PM25GraphDataset(split="train", **common_ds_kwargs)
    val_ds = PM25GraphDataset(split="val", **common_ds_kwargs)

    logger.info("train samples: %d  val samples: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # --- Model ---
    # Derive n_stations from the dataset so exclude_stations is respected.
    # Hardcoding n_stations=18 from config would crash MTGNN when stations are excluded.
    n_stations = train_ds.n_stations
    model = instantiate(cfg.model, n_stations=n_stations, horizons=list(cfg.data.horizons))
    logger.info("Model parameters: %d", model.count_parameters())

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.trainer.lr,
        max_epochs=cfg.trainer.max_epochs,
        patience=cfg.trainer.patience,
        primary_horizon=cfg.trainer.primary_horizon,
        device=device,
        output_dir=Path("."),  # Hydra sets cwd to outputs/{timestamp}
        wandb_project=cfg.trainer.wandb_project,
    )

    history = trainer.fit()

    # --- Summary ---
    best_rmse = min(
        (v for v in history[f"val_rmse_{cfg.trainer.primary_horizon}h"] if v == v),  # skip NaN
        default=float("nan"),
    )
    logger.info("Best val RMSE@%dh = %.4f (normalized scale)", cfg.trainer.primary_horizon, best_rmse)


if __name__ == "__main__":
    main()
