# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Retrain ablation + held-out test evaluation (the decisive Phase 2 analysis).

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Evaluates the Session-8 retrained checkpoints (3-way split: train 2022-23 / val 2024 /
test 2025) on the HELD-OUT test set, in ug/m3, reusing src/training/evaluation.py.

  * ablation_retrained.json - per-horizon test RMSE for the full model and each retrained
    ablation variant (-type_b, -type_c, -adaptive, temporal_only), with the delta vs full.
    This is the decisive test of CRITICAL_REVIEW finding #4 (do the novelties help?).
  * evaluation_test2025.json - headline held-out numbers (persistence / MTGNN / A3TGCN /
    hybrid), the report analogue of evaluation_val2025.json (which stays the pitch artifact).

CRITICAL: each ablation checkpoint is loaded with the SAME gate flags it was trained with
(e.g. no_type_b -> use_type_b=false). The flags do not change the state_dict, so loading a
variant without its flags would silently activate an untrained channel and corrupt the result.

Sanity: the full MTGNN is also scored on VAL (2024) in normalized units; its RMSE@24h must
match the checkpoint's stored val_rmse_24h (Bug-7 denorm guard).

Usage:
    uv run python scripts/11_ablation_eval.py
    uv run python scripts/11_ablation_eval.py device=cpu
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

from src.data.loader import PM25GraphDataset
from src.training.evaluation import (
    build_ground_truth,
    denorm_pred,
    predict,
    rmse_block,
    station_scalers,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_LOCAL_KEYS = {"device", "split", "output_ablation", "output_eval"}

# (name, model_name, flag overrides matching training, checkpoint dir)
_MTGNN_VARIANTS: list[tuple[str, list[str], str]] = [
    ("full", [], "checkpoints_split2/mtgnn"),
    ("no_type_b", ["model.use_type_b=false"], "checkpoints_split2/mtgnn_no_type_b"),
    ("no_type_c", ["model.use_type_c=false"], "checkpoints_split2/mtgnn_no_type_c"),
    ("no_adaptive", ["model.use_adaptive=false"], "checkpoints_split2/mtgnn_no_adaptive"),
    (
        "temporal_only",
        [
            "model.use_type_a=false",
            "model.use_type_b=false",
            "model.use_type_c=false",
            "model.use_adaptive=false",
        ],
        "checkpoints_split2/mtgnn_temporal_only",
    ),
]


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys from Hydra overrides."""
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


def _load_model(
    model_name: str,
    flag_overrides: list[str],
    ckpt_dir: str,
    n_stations: int,
    horizons: list[int],
    base_overrides: list[str],
) -> tuple[torch.nn.Module, dict]:
    """Instantiate a model with the given gate flags and load its checkpoint."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(
            config_name="config",
            overrides=[f"model={model_name}", *flag_overrides, *base_overrides],
        )
    model = instantiate(mcfg.model, n_stations=n_stations, horizons=horizons)
    ckpt = torch.load(
        _PROJECT_ROOT / ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    return model.eval(), ckpt


def _make_dataset(cfg: object, split: str, horizons: list[int], wind_mode: str) -> PM25GraphDataset:
    """Construct a PM25GraphDataset for a split from the composed config."""
    return PM25GraphDataset(
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


def _normalized_rmse_24h(ds: PM25GraphDataset, pred_norm: np.ndarray, horizons: list[int]) -> float:
    """Normalized (pm25_scaled) RMSE@24h for the Bug-7 denorm sanity check."""
    h_idx = horizons.index(24)
    anchors = ds._anchor_indices
    y = ds._pm25_scaled[anchors + 24, :].reshape(-1)
    raw = ds._pm25_raw[anchors + 24, :]
    mask = (
        (~ds._mask_in_loss[anchors + 24, :]) & (~ds._exclude[anchors + 24, :]) & np.isfinite(raw)
    ).reshape(-1)
    err = (pred_norm[:, h_idx] - y)[mask]
    return float(np.sqrt(np.mean(err**2)))


def main() -> None:
    """Evaluate retrained variants on the held-out test set; write ablation + headline JSON."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    test_split = local_args.get("split", "test")
    out_ablation = Path(local_args.get("output_ablation", "outputs/ablation_retrained.json"))
    out_eval = Path(local_args.get("output_eval", "outputs/evaluation_test2025.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")
    n_stations = 18

    test_ds = _make_dataset(cfg, test_split, horizons, wind_mode)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    centers, scales = station_scalers(test_ds)
    y_ug, mask, pers_ug = build_ground_truth(test_ds, horizons)
    persistence_rmse = rmse_block(pers_ug, y_ug, mask, horizons)
    logger.info(
        "Test=%s: %d samples; persistence RMSE=%s", test_split, len(test_ds), persistence_rmse
    )

    # --- Bug-7 sanity: full MTGNN normalized RMSE@24h on VAL must match checkpoint ---
    val_ds = _make_dataset(cfg, "val", horizons, wind_mode)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    full_model, full_ckpt = _load_model(
        "mtgnn", [], "checkpoints_split2/mtgnn", n_stations, horizons, overrides
    )
    val_norm_24h = _normalized_rmse_24h(val_ds, predict(full_model, val_loader, device), horizons)
    ckpt_val = round(float(full_ckpt.get("val_rmse_24h", float("nan"))), 4)
    sane = abs(val_norm_24h - ckpt_val) < 5e-3
    logger.info(
        "DENORM SANITY: val norm RMSE@24h=%.4f vs checkpoint=%.4f -> %s",
        val_norm_24h,
        ckpt_val,
        "PASS" if sane else "WARN",
    )

    # --- Evaluate each MTGNN variant on the held-out test set (ug/m3) ---
    variant_rmse: dict[str, dict[str, float]] = {}
    for name, flags, ckpt_dir in _MTGNN_VARIANTS:
        model, _ = _load_model("mtgnn", flags, ckpt_dir, n_stations, horizons, overrides)
        pred_ug = denorm_pred(predict(model, test_loader, device), centers, scales, n_stations)
        variant_rmse[name] = rmse_block(pred_ug, y_ug, mask, horizons)
        logger.info("test %-14s RMSE=%s", name, variant_rmse[name])

    full_rmse = variant_rmse["full"]
    ablation = {
        name: {
            "rmse_ug_m3": rmse,
            "delta_vs_full": {
                f"{h}h": round(rmse[f"{h}h"] - full_rmse[f"{h}h"], 3) for h in horizons
            },
        }
        for name, rmse in variant_rmse.items()
    }

    # --- A3TGCN on test ---
    a3_model, _ = _load_model(
        "a3tgcn", [], "checkpoints_split2/a3tgcn", n_stations, horizons, overrides
    )
    a3_rmse = rmse_block(
        denorm_pred(predict(a3_model, test_loader, device), centers, scales, n_stations),
        y_ug,
        mask,
        horizons,
    )

    # --- Hybrid operational policy: persistence < 24h, MTGNN (full) >= 24h ---
    full_pred_ug = denorm_pred(
        predict(
            _load_model("mtgnn", [], "checkpoints_split2/mtgnn", n_stations, horizons, overrides)[
                0
            ],
            test_loader,
            device,
        ),
        centers,
        scales,
        n_stations,
    )
    hybrid_ug = pers_ug.copy()
    for h_idx, h in enumerate(horizons):
        if h >= 24:
            hybrid_ug[:, h_idx] = full_pred_ug[:, h_idx]
    hybrid_rmse = rmse_block(hybrid_ug, y_ug, mask, horizons)

    ablation_doc: dict[str, object] = {
        "split": test_split,
        "n_samples": len(test_ds),
        "n_stations": n_stations,
        "horizons_h": horizons,
        "training": "3-way split train 2022-23 / val 2024 / test 2025; single seed per run.",
        "denorm_sanity_val_norm_rmse24h": round(val_norm_24h, 4),
        "denorm_sanity_checkpoint": ckpt_val,
        "persistence_rmse_ug_m3": persistence_rmse,
        "ablation": ablation,
        "caveat": (
            "Each variant is one training run (no fixed seed), so per-channel deltas of a "
            "few tenths of a ug/m3 are within run-to-run variance; read the direction, not "
            "fine ranking. This supersedes the inference-time ablation (08), which was a "
            "lower bound."
        ),
    }
    eval_doc: dict[str, object] = {
        "split": test_split,
        "n_samples": len(test_ds),
        "n_stations": n_stations,
        "horizons_h": horizons,
        "note": "Held-out test 2025 RMSE ug/m3, 3-way split retrain.",
        "persistence_rmse_ug_m3": persistence_rmse,
        "mtgnn": {"checkpoint": "checkpoints_split2/mtgnn", "rmse_ug_m3": full_rmse},
        "a3tgcn": {"checkpoint": "checkpoints_split2/a3tgcn", "rmse_ug_m3": a3_rmse},
        "hybrid_operational": {
            "policy": "persistence for horizons <24h, MTGNN for horizons >=24h",
            "rmse_ug_m3": hybrid_rmse,
        },
    }

    # --- Print ---
    print("\n" + "=" * 74)
    print(f"HELD-OUT TEST ({test_split}) RMSE ug/m3 - 3-way split retrain")
    print("-" * 74)
    print(f"{'method':<22}" + "".join(f"{f'{h}h':>10}" for h in horizons))
    rows = [
        ("persistence", persistence_rmse),
        *[(n, variant_rmse[n]) for n, _, _ in _MTGNN_VARIANTS],
    ]
    rows += [("a3tgcn", a3_rmse), ("hybrid", hybrid_rmse)]
    for label, r in rows:
        print(f"{label:<22}" + "".join(f"{r[f'{h}h']:>10.2f}" for h in horizons))
    print("-" * 74)
    print("ablation delta vs full (+ = removing the channel hurts):")
    for name, _, _ in _MTGNN_VARIANTS:
        if name == "full":
            continue
        d = ablation[name]["delta_vs_full"]
        print(f"  {name:<18}" + "".join(f"{d[f'{h}h']:>+9.3f}" for h in horizons))
    print("=" * 74 + "\n")

    for path, doc in ((out_ablation, ablation_doc), (out_eval, eval_doc)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
