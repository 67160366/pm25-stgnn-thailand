# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""NWP-error sensitivity of the trained MTGNN's accuracy on the 2025 val split.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Addresses reviewer criticism Con#1: ERA5 reanalysis has ~5-7 day latency, so a
production system must use *imperfect NWP forecasts* of the weather drivers at
inference, not perfect reanalysis. This script quantifies how much MTGNN's RMSE
degrades when its five ERA5 inputs (u10, v10, t2m, d2m, blh) are corrupted with
zero-mean Gaussian noise that simulates NWP forecast error.

The noise is injected holistically: the per-feature arrays on the validation
dataset are replaced with perturbed copies before inference, so the corruption
flows into BOTH the station node features (columns 5-9 of x) AND the wind-aware
dynamic graph edges (type_b station→station and type_c hotspot→station, which
read u10/v10 at the anchor when ``wind_mode == "from_field"``). Perturbing only
the node tensor would miss the graph-edge effect.

Denormalisation, the persistence baseline, masking, and per-horizon RMSE all
come from ``src.training.evaluation`` so figures are directly comparable with
``scripts/04_evaluate.py`` (the noise-0 baseline must reproduce MTGNN exactly).

Usage:
    uv run python scripts/06_nwp_sensitivity.py
    uv run python scripts/06_nwp_sensitivity.py device=cpu
    uv run python scripts/06_nwp_sensitivity.py output=outputs/nwp_repro.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
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

_MTGNN_CKPT = "checkpoints/mtgnn/best_model.pt"
_SPLIT_LABEL = "2025-01-01 to 2025-12-31"

# Fractional Gaussian-noise sweep: sigma = fraction * std(feature over val period).
_NOISE_FRACTIONS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
_N_SEEDS = 3

# ERA5 array attributes on PM25GraphDataset, each shape (T_full, N) in physical units.
_ERA5_ATTRS: tuple[str, ...] = ("_u10", "_v10", "_t2m", "_d2m", "_blh")

# Sanity reference: the noise-0 baseline must reproduce scripts/04_evaluate.py MTGNN.
_EXPECTED_CLEAN_RMSE: dict[str, float] = {"6h": 4.31, "12h": 5.80, "24h": 8.67, "48h": 12.68}


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys (device, output) from Hydra config overrides."""
    local_keys = {"device", "output"}
    local: dict[str, str] = {}
    hydra_overrides: list[str] = []
    for arg in argv:
        if "=" not in arg:
            continue
        key, _, val = arg.partition("=")
        if key in local_keys:
            local[key] = val
        else:
            hydra_overrides.append(arg)
    return local, hydra_overrides


def _build_val_dataset(cfg: DictConfig, horizons: list[int], wind_mode: str) -> PM25GraphDataset:
    """Construct the 2025 validation dataset (shared with scripts/04_evaluate.py)."""
    return PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split="val",
        window_in=cfg.data.window_in,
        horizons=horizons,
        exclude_stations=list(cfg.data.exclude_stations),
        graph_config={"wind_mode": wind_mode},
    )


def _load_mtgnn(ds: PM25GraphDataset, horizons: list[int], overrides: list[str]) -> torch.nn.Module:
    """Instantiate MTGNN from the Hydra config and load its trained checkpoint."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])
    model = instantiate(mcfg.model, n_stations=ds.n_stations, horizons=horizons)
    ckpt = torch.load(_PROJECT_ROOT / _MTGNN_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model


def _feature_stds(ds: PM25GraphDataset) -> dict[str, float]:
    """Per-feature std of each clean ERA5 array over the val read range.

    The read range spans every timestep actually consumed during val inference:
    from the earliest input-window start to the latest anchor. This is the natural
    variability used to scale the injected noise.
    """
    lo = int(ds._anchor_indices.min()) - (ds.window_in - 1)
    hi = int(ds._anchor_indices.max())
    stds: dict[str, float] = {}
    for attr in _ERA5_ATTRS:
        block = getattr(ds, attr)[lo : hi + 1, :]
        stds[attr[1:]] = float(np.std(block))
    return stds


def _perturb_era5(
    ds: PM25GraphDataset,
    fraction: float,
    stds: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Replace each ERA5 array with a noisy copy; return originals for restoring.

    Adds independent zero-mean Gaussian noise per element with per-feature
    ``sigma = fraction * stds[feature]``. Because ``__getitem__`` reads these
    arrays live (num_workers=0), the perturbation reaches both the node features
    and the wind-aware graph edges on the next inference pass.

    Args:
        ds: Validation dataset to mutate in place.
        fraction: Noise magnitude as a fraction of each feature's natural std.
        stds: Per-feature std from :func:`_feature_stds`.
        rng: Seeded NumPy generator for reproducibility.

    Returns:
        Mapping of attribute name to the original (clean) array.
    """
    originals: dict[str, np.ndarray] = {}
    for attr in _ERA5_ATTRS:
        clean = getattr(ds, attr)
        originals[attr] = clean
        sigma = fraction * stds[attr[1:]]
        noise = rng.normal(0.0, sigma, size=clean.shape).astype(clean.dtype)
        setattr(ds, attr, clean + noise)
    return originals


def _restore_era5(ds: PM25GraphDataset, originals: dict[str, np.ndarray]) -> None:
    """Restore the clean ERA5 arrays saved by :func:`_perturb_era5`."""
    for attr, arr in originals.items():
        setattr(ds, attr, arr)


def _run_noise_level(
    *,
    fraction: float,
    ds: PM25GraphDataset,
    loader: DataLoader,
    model: torch.nn.Module,
    device: str,
    stds: dict[str, float],
    centers: np.ndarray,
    scales: np.ndarray,
    y_ug: np.ndarray,
    mask: np.ndarray,
    horizons: list[int],
) -> dict[str, float]:
    """Per-horizon RMSE (µg/m³) at one noise level, averaged over seeds.

    Fraction 0.0 runs once on clean ERA5; positive fractions average over
    ``_N_SEEDS`` independent noise realisations, restoring the clean arrays
    between runs.
    """
    if fraction == 0.0:
        pred_ug = denorm_pred(predict(model, loader, device), centers, scales, ds.n_stations)
        return rmse_block(pred_ug, y_ug, mask, horizons)

    per_seed: list[dict[str, float]] = []
    for seed in range(_N_SEEDS):
        rng = np.random.default_rng(seed)
        originals = _perturb_era5(ds, fraction, stds, rng)
        try:
            pred_ug = denorm_pred(predict(model, loader, device), centers, scales, ds.n_stations)
            per_seed.append(rmse_block(pred_ug, y_ug, mask, horizons))
        finally:
            _restore_era5(ds, originals)

    return {key: round(float(np.mean([s[key] for s in per_seed])), 2) for key in per_seed[0]}


def _crossover_fraction(
    results: list[dict[str, Any]], persistence: dict[str, float], horizon_key: str
) -> float | None:
    """Smallest noise fraction at which MTGNN RMSE rises above persistence."""
    for entry in sorted(results, key=lambda e: e["noise_fraction"]):
        if entry["rmse_ug_m3"][horizon_key] > persistence[horizon_key]:
            return float(entry["noise_fraction"])
    return None


def _check_baseline(clean_rmse: dict[str, float]) -> None:
    """Log whether the noise-0 baseline reproduces scripts/04_evaluate.py MTGNN."""
    match = all(abs(clean_rmse[k] - v) < 0.02 for k, v in _EXPECTED_CLEAN_RMSE.items())
    if match:
        logger.info("Noise-0 baseline matches 04_evaluate MTGNN: %s", clean_rmse)
    else:
        logger.warning(
            "Noise-0 baseline DIFFERS from 04_evaluate MTGNN (%s vs %s) — check data/checkpoint.",
            clean_rmse,
            _EXPECTED_CLEAN_RMSE,
        )


def _build_interpretation(
    results: list[dict[str, Any]],
    persistence: dict[str, float],
    crossover: dict[str, float | None],
) -> str:
    """Compose the headline degradation + crossover summary string."""
    clean = results[0]["rmse_ug_m3"]
    worst = results[-1]["rmse_ug_m3"]
    worst_f = results[-1]["noise_fraction"]
    c24, c48 = crossover["24h"], crossover["48h"]
    cross24 = f"{c24:.2f}x std" if c24 is not None else "no tested level (>1.0x std)"
    cross48 = f"{c48:.2f}x std" if c48 is not None else "no tested level (>1.0x std)"
    return (
        "Corrupting all five ERA5 inputs with zero-mean Gaussian noise "
        "(sigma = fraction x per-feature std over the 2025 val period) degrades MTGNN "
        f"monotonically at the long horizons. At the strongest tested noise (fraction="
        f"{worst_f:.2f}), 24h RMSE rises from {clean['24h']:.2f} to {worst['24h']:.2f} ug/m3 and "
        f"48h from {clean['48h']:.2f} to {worst['48h']:.2f}. MTGNN's 24h forecast first exceeds "
        f"the persistence baseline ({persistence['24h']:.2f}) at noise {cross24}; its 48h forecast "
        f"first exceeds persistence ({persistence['48h']:.2f}) at noise {cross48}. The clean 24h "
        f"margin over persistence is razor-thin "
        f"({persistence['24h'] - clean['24h']:.2f} ug/m3), so even modest NWP error erases it, "
        f"whereas the 48h advantage ({persistence['48h'] - clean['48h']:.2f} ug/m3) is more "
        f"robust. Operationally, a "
        "production deployment using imperfect NWP rather than ERA5 reanalysis must keep "
        "weather-feature error well below natural variability to retain MTGNN's edge at 24h."
    )


def _print_table(
    results: list[dict[str, Any]],
    persistence: dict[str, float],
    horizons: list[int],
    crossover: dict[str, float | None],
) -> None:
    """Print the noise-level x horizon RMSE table plus the crossover headline."""
    width = 64
    print("\n" + "=" * width)
    print(f"{'Noise level':<22}" + "".join(f"{f'{h}h':>10}" for h in horizons))
    print("-" * width)
    print(f"{'Persistence':<22}" + "".join(f"{persistence[f'{h}h']:>10.2f}" for h in horizons))
    for entry in results:
        frac = entry["noise_fraction"]
        label = "clean (0.00)" if frac == 0.0 else f"noise {frac:.2f}x std"
        r = entry["rmse_ug_m3"]
        print(f"{label:<22}" + "".join(f"{r[f'{h}h']:>10.2f}" for h in horizons))
    print("=" * width + "  (RMSE µg/m³, lower is better)")
    for hk in ("24h", "48h"):
        c = crossover[hk]
        msg = f"{c:.2f}x std" if c is not None else "not within tested range (<=1.0x std)"
        print(f"  {hk} rises above persistence at noise fraction: {msg}")
    print()


def main() -> None:
    """Run the ERA5-perturbation sensitivity sweep; print + save results."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    output_path = Path(local_args.get("output", "outputs/nwp_sensitivity.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")

    ds = _build_val_dataset(cfg, horizons, wind_mode)
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    logger.info(
        "Val split: %d samples, %d stations, wind_mode=%s", len(ds), ds.n_stations, wind_mode
    )

    centers, scales = station_scalers(ds)
    y_ug, mask, pers_ug = build_ground_truth(ds, horizons)
    persistence_rmse = rmse_block(pers_ug, y_ug, mask, horizons)
    logger.info("Persistence RMSE (µg/m³): %s", persistence_rmse)

    stds = _feature_stds(ds)
    logger.info("ERA5 feature stds (val period): %s", stds)

    model = _load_mtgnn(ds, horizons, overrides)

    results: list[dict[str, Any]] = []
    for fraction in (0.0, *_NOISE_FRACTIONS):
        rmse = _run_noise_level(
            fraction=fraction,
            ds=ds,
            loader=loader,
            model=model,
            device=device,
            stds=stds,
            centers=centers,
            scales=scales,
            y_ug=y_ug,
            mask=mask,
            horizons=horizons,
        )
        beats = {f"{h}h": rmse[f"{h}h"] < persistence_rmse[f"{h}h"] for h in horizons}
        results.append({"noise_fraction": fraction, "rmse_ug_m3": rmse, "beats_persistence": beats})
        logger.info("noise=%.2f RMSE=%s beats=%s", fraction, rmse, beats)

    _check_baseline(results[0]["rmse_ug_m3"])

    crossover: dict[str, float | None] = {
        f"{h}h": _crossover_fraction(results, persistence_rmse, f"{h}h") for h in (24, 48)
    }
    interpretation = _build_interpretation(results, persistence_rmse, crossover)

    output: dict[str, Any] = {
        "meta": {
            "val_split": _SPLIT_LABEL,
            "n_stations": ds.n_stations,
            "model": "mtgnn",
            "n_seeds": _N_SEEDS,
            "feature_stds": {k: round(v, 4) for k, v in stds.items()},
        },
        "noise_design": (
            "Per-element zero-mean Gaussian noise added to all five ERA5 arrays "
            "(u10, v10, t2m, d2m, blh); per-feature sigma = noise_fraction * std(feature "
            "over val period). Perturbs node features AND wind-aware graph edges. RMSE in "
            "ug/m3, averaged over n_seeds realisations per noise level."
        ),
        "persistence_rmse_ug_m3": persistence_rmse,
        "results": results,
        "crossover_fraction": crossover,
        "interpretation": interpretation,
    }

    _print_table(results, persistence_rmse, horizons, crossover)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    logger.info("Saved sensitivity analysis to %s", output_path)


if __name__ == "__main__":
    main()
