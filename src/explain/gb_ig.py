# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Explainability methods for PM2.5 STGNN source attribution.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Implements two attribution methods for the heterogeneous PM2.5 graph:

1. ``integrated_gradients`` — path-integral attribution over station node
   features from a zero baseline to the actual input (Sundararajan et al. 2017).
   Satisfies the completeness axiom: sum of attributions equals the difference
   in model output between the input and the baseline.

2. ``occlusion_country_attribution`` — model-agnostic country-level source
   attribution by masking the total_frp of hotspot nodes per country and
   measuring the resulting drop in the predicted PM2.5 value.
   This is the primary method for the NSC "source attribution" novelty.

References:
    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for
    Deep Networks. ICML 2017. https://proceedings.mlr.press/v70/sundararajan17a
"""

from __future__ import annotations

import logging

import torch
from torch_geometric.data import HeteroData

from src.models.base import PM25ModelBase

logger = logging.getLogger(__name__)


def integrated_gradients(
    model: PM25ModelBase,
    data: HeteroData,
    target_station_idx: int,
    target_horizon_idx: int,
    n_steps: int = 50,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Compute Integrated Gradients attributions for station node features.

    Approximates the path integral of gradients from a zero baseline to the
    actual input using a Riemann sum with ``n_steps`` steps. The result
    satisfies the completeness axiom: sum(attr) = f(x) - f(baseline).

    Args:
        model: Trained PM25ModelBase instance (eval mode used internally).
        data: Single-sample HeteroData (un-batched). Station x shape (N, T_in, F).
        target_station_idx: Index in [0, N) identifying the station to explain.
        target_horizon_idx: Index in [0, H) identifying the forecast horizon.
        n_steps: Riemann sum steps (50 is sufficient; 100 for publication quality).
        device: Torch device for computation.

    Returns:
        Dict with single key ``"station_x"``: FloatTensor of shape (N, T_in, F).
        Positive values indicate features that increase the prediction; negative
        values indicate features that suppress it.
    """
    device = torch.device(device)
    model = model.eval().to(device)
    data = data.to(device)

    orig_x = data["station"].x.detach()  # (N, T_in, F)
    baseline = torch.zeros_like(orig_x)
    integrated_grads = torch.zeros_like(orig_x)

    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        x_alpha = (baseline + alpha * (orig_x - baseline)).requires_grad_(True)
        data["station"].x = x_alpha

        pred = model(data)  # (N, H)
        scalar = pred[target_station_idx, target_horizon_idx]
        scalar.backward()

        with torch.no_grad():
            if x_alpha.grad is not None:
                integrated_grads += x_alpha.grad

        model.zero_grad()

    data["station"].x = orig_x  # restore

    integrated_grads /= n_steps
    attr = (orig_x - baseline) * integrated_grads

    with torch.no_grad():
        f_x = model(data)[target_station_idx, target_horizon_idx].item()
        data["station"].x = baseline
        f_baseline = model(data)[target_station_idx, target_horizon_idx].item()
        data["station"].x = orig_x  # restore after baseline forward

    logger.debug(
        "IG completeness check: attr_sum=%.4f  delta_f=%.4f",
        attr.sum().item(),
        f_x - f_baseline,
    )

    return {"station_x": attr.detach().cpu()}


def _forward_with_hotspot_x(
    model: PM25ModelBase, data: HeteroData, hx: torch.Tensor
) -> torch.Tensor:
    """Forward pass with ``data["hotspot"].x`` swapped for ``hx``, state always restored.

    The single place where a counterfactual hotspot tensor reaches the model, so
    every "what-if" in this module runs through the *same* unmodified
    ``model(data)`` call as the real forecast — there is no branch anywhere that
    inspects the country label, the date, or the size of the edit.

    Args:
        model: Trained PM25ModelBase (already on the target device, eval mode).
        data: Single-sample HeteroData (already on the target device).
        hx: Replacement hotspot feature tensor, same shape as ``data["hotspot"].x``.

    Returns:
        Detached ``(N, H)`` prediction grid in the model's normalised scale.
    """
    orig_hx = data["hotspot"].x
    data["hotspot"].x = hx
    try:
        with torch.no_grad():
            return model(data).detach().clone()
    finally:
        data["hotspot"].x = orig_hx  # restore


def _occlude_country_forward(
    model: PM25ModelBase,
    data: HeteroData,
    hotspot_countries: list[str],
    country: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Forward pass with one country's hotspot features zeroed, state restored.

    Zeroes every feature (frp, lat, lon) of the hotspot nodes labelled ``country``
    and returns the full prediction grid. ``data["hotspot"].x`` is always restored
    to its original value before returning (even on error).

    Args:
        model: Trained PM25ModelBase (already on ``device``, eval mode).
        data: Single-sample HeteroData (already on ``device``).
        hotspot_countries: Country label per hotspot node.
        country: Country to occlude.
        device: Torch device.

    Returns:
        Tuple ``(pred, n_occluded)`` where ``pred`` is the ``(N, H)`` prediction
        with the country removed and ``n_occluded`` is the number of nodes zeroed.
    """
    mask = torch.tensor([c == country for c in hotspot_countries], dtype=torch.bool, device=device)
    hx_occluded = data["hotspot"].x.clone()
    hx_occluded[mask] = 0.0
    pred = _forward_with_hotspot_x(model, data, hx_occluded)
    return pred, int(mask.sum().item())


def occlusion_country_attribution(
    model: PM25ModelBase,
    data: HeteroData,
    target_station_idx: int,
    target_horizon_idx: int,
    hotspot_countries: list[str],
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Attribute PM2.5 prediction to pollution source countries via occlusion.

    For each country present in the hotspot set, zeroes out the total_frp of
    all hotspot nodes from that country and measures the resulting drop in the
    predicted PM2.5 value. A larger drop means stronger contribution.

    This method is model-agnostic (no gradient computation) and always produces
    non-negative, interpretable scores. Country labels come from the FIRMS
    hotspot clustering bbox geocoder (Thailand / Myanmar / Laos / other).

    Args:
        model: Trained PM25ModelBase instance.
        data: Single-sample HeteroData with hotspot.x shape (n_hotspots, 3)
              where column 0 is total_frp.
        target_station_idx: Station to explain (index in [0, N)).
        target_horizon_idx: Forecast horizon to explain (index in [0, H)).
        hotspot_countries: Country label for each hotspot node. Length must
            equal the number of hotspot nodes in ``data``.
        device: Torch device for computation.

    Returns:
        Dict mapping country name to normalized attribution score in [0, 1].
        Scores sum to 1.0. Countries with no attributable influence are omitted.
        Returns empty dict if there are no hotspot nodes.
    """
    if not hotspot_countries:
        return {}

    device = torch.device(device)
    model = model.eval().to(device)
    data = data.to(device)

    with torch.no_grad():
        pred_full = model(data)[target_station_idx, target_horizon_idx].item()

    countries = sorted(set(hotspot_countries))
    raw_attr: dict[str, float] = {}

    for country in countries:
        pred, _ = _occlude_country_forward(model, data, hotspot_countries, country, device)
        pred_occluded = pred[target_station_idx, target_horizon_idx].item()
        raw_attr[country] = max(0.0, pred_full - pred_occluded)

    total = sum(raw_attr.values())
    if total > 1e-8:
        return {c: v / total for c, v in raw_attr.items()}
    return {c: 0.0 for c in countries}


def counterfactual_occlusion(
    model: PM25ModelBase,
    data: HeteroData,
    country: str,
    hotspot_countries: list[str],
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """What-if: remove one country's fires and return the whole prediction grid.

    Occludes (zeroes the features of) every hotspot node labelled ``country`` and
    contrasts the model's full prediction with the occluded one across *all*
    stations and horizons — the interactive "policy lever" behind the dashboard
    counterfactual ("if these fires were put out, how much would PM2.5 drop?").

    Predictions are returned in the model's normalised scale; the caller
    denormalises per station via ``src.training.evaluation.denorm_pred``. The
    per-station drop in µg/m³ is ``delta_norm * scale`` (the RobustScaler centre
    cancels in the difference), so denormalising both grids and subtracting is
    equivalent and numerically stable.

    This is a *what-if under the model*, not a validated causal claim: the
    magnitude depends entirely on the trained model's learned sensitivity.

    Args:
        model: Trained PM25ModelBase.
        data: Single-sample HeteroData with ``hotspot.x`` (n_hotspots, 3).
        country: Country whose hotspot nodes to occlude.
        hotspot_countries: Country label per hotspot node.
        device: Torch device.

    Returns:
        Dict with:
            - ``"pred_full"``: FloatTensor (N, H) — unmodified prediction.
            - ``"pred_occluded"``: FloatTensor (N, H) — with ``country`` removed.
            - ``"delta"``: FloatTensor (N, H) = ``pred_full - pred_occluded``
              (positive = removing the country lowers predicted PM2.5).
            - ``"country"``: the requested country.
            - ``"n_occluded"``: number of hotspot nodes zeroed (0 = no-op).
            - ``"available_countries"``: sorted unique labels in the sample.

        When the sample has no hotspots or ``country`` is absent, ``pred_occluded``
        equals ``pred_full`` and ``delta`` is all zeros (a safe no-op).
    """
    device = torch.device(device)
    model = model.eval().to(device)
    data = data.to(device)

    with torch.no_grad():
        pred_full = model(data).detach().clone()  # (N, H)

    available = sorted(set(hotspot_countries))
    if hotspot_countries and country in available:
        pred_occluded, n_occluded = _occlude_country_forward(
            model, data, hotspot_countries, country, device
        )
    else:
        pred_occluded, n_occluded = pred_full.clone(), 0

    delta = pred_full - pred_occluded
    return {
        "pred_full": pred_full.cpu(),
        "pred_occluded": pred_occluded.cpu(),
        "delta": delta.cpu(),
        "country": country,
        "n_occluded": n_occluded,
        "available_countries": available,
    }


# ---------------------------------------------------------------------------
# Node-level counterfactuals ("switch off *these* fires")
# ---------------------------------------------------------------------------


def connected_hotspot_indices(data: HeteroData, station_idx: int) -> list[int]:
    """Hotspot node indices wired to one station by a Type-C (wind-aligned) edge.

    Type-C edges exist only where the graph builder found the fire within
    ``type_c_max_km`` *and* the wind pointing from the fire toward the station, so
    this is the model's own answer to "which fires can reach here today" — read
    off the graph, not recomputed with a second heuristic.

    Args:
        data: Single-sample HeteroData carrying ``("hotspot","type_c","station")``.
        station_idx: Station node index in ``[0, N)``.

    Returns:
        Sorted unique hotspot indices with an edge into ``station_idx`` (may be empty).
    """
    store = data["hotspot", "type_c", "station"]
    ei = getattr(store, "edge_index", None)
    if ei is None or ei.numel() == 0:
        return []
    src, dst = ei[0], ei[1]
    return sorted({int(i) for i in src[dst == station_idx].tolist()})


def occlude_hotspot_nodes(
    model: PM25ModelBase,
    data: HeteroData,
    node_indices: list[int],
    remaining_fraction: float = 0.0,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """What-if: suppress *specific* fire clusters and re-run the same model.

    Scales ``total_frp`` (hotspot feature column 0) of the selected nodes by
    ``remaining_fraction`` and contrasts the model's prediction with the
    unmodified one across all stations and horizons. This is the map-click
    counterpart of :func:`counterfactual_occlusion`: the user picks the nodes
    instead of a country label.

    Deliberately differs from :func:`counterfactual_occlusion`, which zeroes the
    *whole* feature row (frp, lat, lon) for one country and whose output is frozen
    into the report JSONs. Here lat/lon are left intact, because a fire that has
    been put out is a zero-intensity fire *at its own location*, not a phantom
    node teleported to (0, 0) — and because partial suppression (0 < fraction < 1)
    is only meaningful for intensity. Do not "harmonise" the two: the country
    function's behaviour is load-bearing for reproducing published numbers.

    The country label is not touched and cannot be: ``hotspot.x`` is
    ``[total_frp, lat, lon]`` (see ``src/data/graph_builder.build_graph``), so the
    network never sees which country a fire is in. Any country-level effect is
    emergent from fire location, intensity and the wind-built Type-C edges.

    Predictions are returned in the model's normalised scale; the caller
    denormalises per station via ``src.training.evaluation.denorm_pred``.

    This is a *what-if under the model*, not a validated causal claim: the
    magnitude depends entirely on the trained model's learned sensitivity.

    Args:
        model: Trained PM25ModelBase.
        data: Single-sample HeteroData with ``hotspot.x`` (n_hotspots, 3).
        node_indices: Hotspot node indices to suppress. Out-of-range and duplicate
            entries are ignored; an empty selection is a safe no-op.
        remaining_fraction: Fraction of the original FRP left burning. ``0.0``
            fully extinguishes, ``0.5`` halves, ``1.0`` changes nothing.
        device: Torch device.

    Returns:
        Dict with:
            - ``"pred_full"``: FloatTensor (N, H) — unmodified prediction.
            - ``"pred_occluded"``: FloatTensor (N, H) — with the selection suppressed.
            - ``"delta"``: FloatTensor (N, H) = ``pred_full - pred_occluded``
              (positive = suppressing these fires lowers predicted PM2.5).
            - ``"x_before"`` / ``"x_after"``: FloatTensor (k, 3) — the selected
              hotspot feature rows as the model actually saw them, for display.
            - ``"node_indices"``: the cleaned, sorted selection.
            - ``"n_selected"``: ``k``, the number of nodes edited.
            - ``"frp_removed"``: total MW of fire radiative power taken out.
            - ``"remaining_fraction"``: echo of the requested fraction.
    """
    device = torch.device(device)
    model = model.eval().to(device)
    data = data.to(device)

    orig_hx = data["hotspot"].x
    n_nodes = int(orig_hx.shape[0])
    sel = sorted({int(i) for i in node_indices if 0 <= int(i) < n_nodes})

    pred_full = _forward_with_hotspot_x(model, data, orig_hx)

    if not sel:
        empty = orig_hx.new_zeros((0, orig_hx.shape[1]))
        return {
            "pred_full": pred_full.cpu(),
            "pred_occluded": pred_full.clone().cpu(),
            "delta": torch.zeros_like(pred_full).cpu(),
            "x_before": empty.cpu(),
            "x_after": empty.cpu(),
            "node_indices": [],
            "n_selected": 0,
            "frp_removed": 0.0,
            "remaining_fraction": float(remaining_fraction),
        }

    idx = torch.tensor(sel, dtype=torch.long, device=device)
    hx = orig_hx.clone()
    hx[idx, 0] = hx[idx, 0] * float(remaining_fraction)
    pred_occluded = _forward_with_hotspot_x(model, data, hx)

    frp_removed = float((orig_hx[idx, 0] - hx[idx, 0]).sum().item())
    return {
        "pred_full": pred_full.cpu(),
        "pred_occluded": pred_occluded.cpu(),
        "delta": (pred_full - pred_occluded).cpu(),
        "x_before": orig_hx[idx].clone().cpu(),
        "x_after": hx[idx].clone().cpu(),
        "node_indices": sel,
        "n_selected": len(sel),
        "frp_removed": frp_removed,
        "remaining_fraction": float(remaining_fraction),
    }


def dose_response(
    model: PM25ModelBase,
    data: HeteroData,
    node_indices: list[int],
    fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0),
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Sweep the suppression level of one fire selection and record the response curve.

    Runs :func:`occlude_hotspot_nodes` once per entry in ``fractions``. The point
    of the sweep is falsifiability: a hand-written rule that "subtracts X when
    Myanmar fires are present" produces a step, while a network responding to the
    FRP feature produces a smooth, monotone curve. The caller plots it as-is.

    Args:
        model: Trained PM25ModelBase.
        data: Single-sample HeteroData.
        node_indices: Hotspot nodes to suppress.
        fractions: Remaining-FRP fractions to evaluate, in the order plotted.
        device: Torch device.

    Returns:
        Dict with ``"fractions"`` (list, echoed) and ``"pred"``: FloatTensor
        ``(len(fractions), N, H)`` of normalised predictions, one grid per level.
    """
    grids = [
        occlude_hotspot_nodes(model, data, node_indices, f, device=device)["pred_occluded"]
        for f in fractions
    ]
    return {"fractions": list(fractions), "pred": torch.stack(grids)}


def matched_placebo_indices(
    data: HeteroData,
    node_indices: list[int],
    station_idx: int,
    max_nodes: int | None = None,
) -> list[int]:
    """Pick a control set of fires the wind does *not* connect to the station.

    The negative control for a map-click counterfactual: same station, same day,
    a comparable amount of burning — but taken from clusters with no Type-C edge
    into ``station_idx``. If the model is reading the graph rather than reacting
    to "fire exists anywhere", suppressing this set should move the forecast far
    less than suppressing the real selection.

    Greedy by descending FRP until the removed power reaches the real selection's
    total, so the control is matched on the quantity that actually feeds the model
    (FRP), not merely on node count.

    Args:
        data: Single-sample HeteroData with ``hotspot.x`` (n_hotspots, 3).
        node_indices: The real selection being controlled for.
        station_idx: Station the real selection was chosen against.
        max_nodes: Optional cap on the control set size.

    Returns:
        Sorted hotspot indices of the control set; empty when the day has no
        unconnected fires to draw on.
    """
    hx = data["hotspot"].x
    if hx.numel() == 0:
        return []
    connected = set(connected_hotspot_indices(data, station_idx))
    selected = {int(i) for i in node_indices}
    target_frp = float(sum(float(hx[i, 0]) for i in selected if 0 <= i < hx.shape[0]))

    pool = [i for i in range(int(hx.shape[0])) if i not in connected and i not in selected]
    pool.sort(key=lambda i: -float(hx[i, 0]))

    chosen: list[int] = []
    total = 0.0
    for i in pool:
        if total >= target_frp or (max_nodes is not None and len(chosen) >= max_nodes):
            break
        chosen.append(i)
        total += float(hx[i, 0])
    return sorted(chosen)
