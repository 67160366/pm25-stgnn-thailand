# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Inference-time channel ablation of the trained MTGNN (quick diagnostic).

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

The three novelties (wind-aware type_b edges, hotspot type_c bipartite edges, and the
self-learned adaptive adjacency) are the project's claimed contribution, yet nothing yet
shows they improve accuracy. This script gives a *cheap* first signal using the EXISTING
checkpoint, with no retraining: at inference we knock out one spatial channel at a time and
measure the change in val RMSE per horizon.

    * -type_b   : empty the type_b edge_index (no wind-graph message passing).
    * -hotspot  : zero the hotspot features and empty type_c (model sets c = 0 exactly).
    * -adaptive : reversibly zero ``adaptive_proj`` weight+bias (g = 0), restored after.
    * -all_novelties : all three at once.

A near-zero (or negative) RMSE delta means the trained model barely uses that channel.

IMPORTANT CAVEAT (also written into the output JSON): this measures the *trained model's
sensitivity* to removing a channel at inference, NOT the value of having trained with it.
A network can route information around a disabled channel, so this is a LOWER-BOUND
diagnostic only. The decisive test is the retrain ablation (Phase 2.2 / ablation_retrained).

Denormalisation, masking and the persistence baseline reuse ``src/training/evaluation.py``.

Usage:
    uv run python scripts/08_inference_ablation.py
    uv run python scripts/08_inference_ablation.py split=test output=outputs/ablation_inf_test.json
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from src.data.loader import PM25GraphDataset
from src.training.evaluation import (
    build_ground_truth,
    denorm_pred,
    rmse_block,
    station_scalers,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_MTGNN_CKPT = "checkpoints/mtgnn/best_model.pt"
_LOCAL_KEYS = {"device", "output", "split"}

_CAVEAT = (
    "Inference-time channel removal on the trained checkpoint (NO retrain). This measures "
    "the trained model's sensitivity to disabling a channel at inference, NOT the value of "
    "training with it (the network may route around a disabled channel). Treat deltas as a "
    "LOWER BOUND on each channel's importance; the decisive test is the retrain ablation."
)

BatchMutator = Callable[[HeteroData], HeteroData]


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys (device/output/split) from Hydra overrides."""
    local: dict[str, str] = {}
    overrides: list[str] = []
    for arg in argv:
        if "=" not in arg:
            continue
        key, _, val = arg.partition("=")
        if key in _LOCAL_KEYS:
            local[key] = val
        else:
            overrides.append(arg)
    return local, overrides


def _load_mtgnn(ds: PM25GraphDataset, horizons: list[int], overrides: list[str]) -> torch.nn.Module:
    """Instantiate MTGNN from Hydra config and load the trained checkpoint."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])
    model = instantiate(mcfg.model, n_stations=ds.n_stations, horizons=horizons)
    ckpt = torch.load(_PROJECT_ROOT / _MTGNN_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model


def _empty_edges(batch: HeteroData, edge_type: tuple[str, str, str]) -> None:
    """Replace an edge store with zero edges (preserving attr width), in place."""
    store = batch[edge_type]
    dev = store.edge_index.device
    k = store.edge_attr.size(1)
    store.edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
    store.edge_attr = torch.zeros((0, k), dtype=store.edge_attr.dtype, device=dev)


def _mutate_no_type_b(batch: HeteroData) -> HeteroData:
    """Remove wind-aware message passing by emptying type_b edges."""
    _empty_edges(batch, ("station", "type_b", "station"))
    return batch


def _mutate_no_hotspot(batch: HeteroData) -> HeteroData:
    """Remove the fire signal: zero hotspot features and empty type_c edges (-> c = 0)."""
    hx = batch["hotspot"].x
    batch["hotspot"].x = torch.zeros((0, hx.size(1)), dtype=hx.dtype, device=hx.device)
    _empty_edges(batch, ("hotspot", "type_c", "station"))
    return batch


def _mutate_no_all(batch: HeteroData) -> HeteroData:
    """Remove both edge-based novelties (adaptive is handled via param zeroing)."""
    return _mutate_no_hotspot(_mutate_no_type_b(batch))


# name -> (batch mutator or None, zero the adaptive_proj term?)
_VARIANTS: list[tuple[str, BatchMutator | None, bool]] = [
    ("full", None, False),
    ("no_type_b", _mutate_no_type_b, False),
    ("no_hotspot", _mutate_no_hotspot, False),
    ("no_adaptive", None, True),
    ("no_all_novelties", _mutate_no_all, True),
]


def _predict_variant(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    mutate: BatchMutator | None,
    zero_adaptive: bool,
) -> np.ndarray:
    """Run inference with an optional per-batch mutation and optional adaptive knock-out."""
    saved: tuple[torch.Tensor, torch.Tensor] | None = None
    if zero_adaptive:
        saved = (
            model.adaptive_proj.weight.data.clone(),
            model.adaptive_proj.bias.data.clone(),
        )
        model.adaptive_proj.weight.data.zero_()
        model.adaptive_proj.bias.data.zero_()

    model.eval().to(device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if mutate is not None:
                batch = mutate(batch)
            preds.append(model(batch).detach().cpu().numpy())

    if saved is not None:
        model.adaptive_proj.weight.data.copy_(saved[0])
        model.adaptive_proj.bias.data.copy_(saved[1])
    return np.concatenate(preds, axis=0)


def main() -> None:
    """Run inference-time channel ablations; print + save per-horizon RMSE deltas."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    split = local_args.get("split", "val")
    output_path = Path(local_args.get("output", "outputs/ablation_inference.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")

    ds = PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split=split,
        window_in=cfg.data.window_in,
        horizons=horizons,
        exclude_stations=list(cfg.data.exclude_stations),
        graph_config={"wind_mode": wind_mode},
    )
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    logger.info("Split=%s: %d samples, %d stations", split, len(ds), ds.n_stations)

    centers, scales = station_scalers(ds)
    y_ug, mask, pers_ug = build_ground_truth(ds, horizons)
    persistence_rmse = rmse_block(pers_ug, y_ug, mask, horizons)
    model = _load_mtgnn(ds, horizons, overrides)

    variants: dict[str, dict[str, float]] = {}
    full_rmse: dict[str, float] = {}
    for name, mutate, zero_adaptive in _VARIANTS:
        pred_ug = denorm_pred(
            _predict_variant(model, loader, device, mutate, zero_adaptive),
            centers,
            scales,
            ds.n_stations,
        )
        rmse = rmse_block(pred_ug, y_ug, mask, horizons)
        if name == "full":
            full_rmse = rmse
        delta = {f"{h}h": round(rmse[f"{h}h"] - full_rmse[f"{h}h"], 3) for h in horizons}
        variants[name] = {"rmse_ug_m3": rmse, "delta_vs_full": delta}
        logger.info("%-16s rmse=%s delta=%s", name, rmse, delta)

    results: dict[str, object] = {
        "split": split,
        "n_stations": ds.n_stations,
        "n_samples": len(ds),
        "horizons_h": horizons,
        "caveat": _CAVEAT,
        "persistence_rmse_ug_m3": persistence_rmse,
        "variants": variants,
        "interpretation": (
            "delta_vs_full > 0 means removing the channel hurts (the trained model uses it); "
            "delta ~ 0 means the channel is barely used at inference. This is a lower bound; "
            "see ablation_retrained.json for the decisive retrain ablation."
        ),
    }

    print("\n" + "=" * 70)
    print(f"{'Variant':<18}" + "".join(f"{f'{h}h':>9}" for h in horizons) + "   (RMSE ug/m3)")
    print("-" * 70)
    for name in variants:
        r = variants[name]["rmse_ug_m3"]
        print(f"{name:<18}" + "".join(f"{r[f'{h}h']:>9.2f}" for h in horizons))
    print("-" * 70)
    print(f"{'delta vs full:':<18}")
    for name in variants:
        if name == "full":
            continue
        d = variants[name]["delta_vs_full"]
        print(f"{name:<18}" + "".join(f"{d[f'{h}h']:>+9.3f}" for h in horizons))
    print("=" * 70 + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved inference-ablation results to %s", output_path)


if __name__ == "__main__":
    main()
