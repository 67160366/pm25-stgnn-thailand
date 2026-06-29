# [TODO: NSC Disclaimer - see booklet page 44]

"""Multi-seed retrain ablation: per-channel contribution vs run-to-run variance.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

The single-seed ablation (`11_ablation_eval.py`) could not tell a real per-channel effect from
training noise. Here we evaluate EACH variant's three retrained checkpoints (the original run in
`checkpoints_split2/` + two seeded runs in `checkpoints_split2_seeds/`) on the held-out test set,
and report per-horizon RMSE mean +/- std. A channel's effect is called robust only if the mean
delta vs full exceeds the combined (full + variant) std at that horizon.

Each variant is loaded with the SAME gate flags it was trained with. Reuses
`src/training/evaluation.py`. Output: `outputs/ablation_multiseed.json`.

Usage:
    uv run python scripts/12_ablation_multiseed.py
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
_LOCAL_KEYS = {"device", "split", "output"}
_BASE_DIR = "checkpoints_split2"
_SEEDS_DIR = "checkpoints_split2_seeds"

# name -> (gate-flag overrides, original-run dir name under checkpoints_split2/)
_VARIANTS: dict[str, tuple[list[str], str]] = {
    "full": ([], "mtgnn"),
    "no_type_b": (["model.use_type_b=false"], "mtgnn_no_type_b"),
    "no_type_c": (["model.use_type_c=false"], "mtgnn_no_type_c"),
    "no_adaptive": (["model.use_adaptive=false"], "mtgnn_no_adaptive"),
    "temporal_only": (
        [
            "model.use_type_a=false",
            "model.use_type_b=false",
            "model.use_type_c=false",
            "model.use_adaptive=false",
        ],
        "mtgnn_temporal_only",
    ),
}


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


def _ckpt_paths(name: str, base: str) -> list[str]:
    """Existing checkpoint dirs for a variant: original run + seeded runs."""
    candidates = [f"{_BASE_DIR}/{base}"] + [f"{_SEEDS_DIR}/{name}_s{s}" for s in (0, 1)]
    return [p for p in candidates if (_PROJECT_ROOT / p / "best_model.pt").exists()]


def _eval_ckpt(
    flags: list[str],
    ckpt_dir: str,
    n_stations: int,
    horizons: list[int],
    base_overrides: list[str],
    loader: DataLoader,
    device: str,
    scalers: tuple[np.ndarray, np.ndarray],
    y_ug: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Load one checkpoint with its gate flags and return per-horizon test RMSE (ug/m3)."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(config_name="config", overrides=["model=mtgnn", *flags, *base_overrides])
    model = instantiate(mcfg.model, n_stations=n_stations, horizons=horizons)
    state = torch.load(
        _PROJECT_ROOT / ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)
    pred_ug = denorm_pred(predict(model.eval(), loader, device), scalers[0], scalers[1], n_stations)
    return rmse_block(pred_ug, y_ug, mask, horizons)


def _aggregate(rmses: list[dict[str, float]], horizons: list[int]) -> dict[str, list[float]]:
    """Mean and sample-std per horizon over a variant's seeded runs."""
    vals = np.array([[r[f"{h}h"] for h in horizons] for r in rmses], dtype=np.float64)
    std = vals.std(axis=0, ddof=1) if vals.shape[0] > 1 else np.zeros(len(horizons))
    return {
        "mean": [round(float(m), 3) for m in vals.mean(axis=0)],
        "std": [round(float(s), 3) for s in std],
    }


def main() -> None:
    """Evaluate every variant's seeded checkpoints on test; report mean +/- std + robustness."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    split = local_args.get("split", "test")
    output_path = Path(local_args.get("output", "outputs/ablation_multiseed.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")
    n_stations = 18

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
    scalers = station_scalers(ds)
    y_ug, mask, pers_ug = build_ground_truth(ds, horizons)
    persistence = rmse_block(pers_ug, y_ug, mask, horizons)

    agg: dict[str, dict] = {}
    for name, (flags, base) in _VARIANTS.items():
        paths = _ckpt_paths(name, base)
        rmses = [
            _eval_ckpt(
                flags, p, n_stations, horizons, overrides, loader, device, scalers, y_ug, mask
            )
            for p in paths
        ]
        agg[name] = {"n_seeds": len(rmses), **_aggregate(rmses, horizons)}
        logger.info(
            "%-14s n=%d mean=%s std=%s", name, len(rmses), agg[name]["mean"], agg[name]["std"]
        )

    full = agg["full"]
    deltas: dict[str, dict] = {}
    for name in _VARIANTS:
        if name == "full":
            continue
        var = agg[name]
        delta = [round(var["mean"][i] - full["mean"][i], 3) for i in range(len(horizons))]
        noise = [round(var["std"][i] + full["std"][i], 3) for i in range(len(horizons))]
        robust = [abs(delta[i]) > noise[i] for i in range(len(horizons))]
        deltas[name] = {
            "delta_vs_full": delta,
            "combined_std": noise,
            "robust_beyond_noise": robust,
        }

    results: dict[str, object] = {
        "split": split,
        "n_samples": len(ds),
        "horizons_h": horizons,
        "seeds_per_variant": "original run + seeds 0,1 (up to 3)",
        "persistence_rmse_ug_m3": persistence,
        "variants_mean_std_ug_m3": agg,
        "deltas_vs_full": deltas,
        "interpretation": (
            "A channel's contribution is robust only where robust_beyond_noise is true (|mean "
            "delta vs full| exceeds full.std + variant.std). Otherwise the single-seed delta was "
            "within training noise."
        ),
    }

    print("\n" + "=" * 78)
    print(f"MULTI-SEED ablation on {split} (RMSE ug/m3, mean +/- std over seeds)")
    print("-" * 78)
    print(f"{'variant':<16}" + "".join(f"{f'{h}h':>15}" for h in horizons))
    for name in _VARIANTS:
        m, s = agg[name]["mean"], agg[name]["std"]
        print(f"{name:<16}" + "".join(f"{m[i]:>8.2f}+-{s[i]:<5.2f}" for i in range(len(horizons))))
    print("-" * 78)
    print("delta vs full (robust beyond seed noise marked *):")
    for name, d in deltas.items():
        cells = "".join(
            f"{d['delta_vs_full'][i]:>+8.2f}{'*' if d['robust_beyond_noise'][i] else ' '}    "
            for i in range(len(horizons))
        )
        print(f"  {name:<14}{cells}")
    print("=" * 78 + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
