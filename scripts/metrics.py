from typing import Callable

import numpy as np

DELTA = 1e-6
EPS = 1e-9


def _midpoint(minmax: np.ndarray) -> np.ndarray:
    return 0.5 * (minmax[:, 0] + minmax[:, 1])


def _range(minmax: np.ndarray) -> np.ndarray:
    return minmax[:, 1] - minmax[:, 0]


Floating = np.ndarray | float
def _safe_div(a: Floating, b: Floating, eps: float = EPS) -> Floating:
    return a / (b + eps)


# ================================================== REACTION METRICS ==================================================


def minimum(minmax: np.ndarray) -> float:
    return float(np.min(minmax[:, 0]))


def maximum(minmax: np.ndarray) -> float:
    return float(np.max(minmax[:, 1]))


def mean_range(minmax: np.ndarray) -> float:
    r = _range(minmax)
    return float(np.mean(r))


def mean_midpoint(minmax: np.ndarray) -> float:
    mid = _midpoint(minmax)
    return float(np.mean(mid))


def mean_relative_range(minmax: np.ndarray) -> float:
    r = _range(minmax)
    cap = np.maximum(np.abs(minmax[:, 0]), np.abs(minmax[:, 1]))
    return float(np.mean(_safe_div(r, cap)))


def median_range(minmax: np.ndarray) -> float:
    r = _range(minmax)
    return float(np.median(r))


def median_midpoint(minmax: np.ndarray) -> float:
    mid = _midpoint(minmax)
    return float(np.median(mid))


def box_range(minmax: np.ndarray) -> float:
    r = _range(minmax)
    return float(np.percentile(r, 75) - np.percentile(r, 25))


def frac_variable(minmax: np.ndarray, delta: float = DELTA) -> float:
    r = _range(minmax)
    return float(np.mean(r > delta))


def frac_zero_fixed(minmax: np.ndarray, delta: float = DELTA) -> float:
    return float(np.mean((np.abs(minmax[:, 0]) <= delta) & (np.abs(minmax[:, 1]) <= delta)))


def frac_bidirectional(minmax: np.ndarray, delta: float = DELTA) -> float:
    return float(np.mean((minmax[:, 0] < -delta) & (minmax[:, 1] > delta)))


def mean_abs_flux(minmax: np.ndarray) -> float:
    cap = np.maximum(np.abs(minmax[:, 0]), np.abs(minmax[:, 1]))
    return float(np.mean(cap))


def std_range(minmax: np.ndarray) -> float:
    r = _range(minmax)
    return float(np.std(r))


def median_midpoint_over_range_norm(minmax: np.ndarray, eps: float = EPS) -> float:
    mid = np.abs(_midpoint(minmax))
    r = _range(minmax)
    val = _safe_div(mid, r + eps)
    return float(np.median(val))


REACTION_METRIC_LIST = [
    minimum,
    maximum,
    mean_range,
    mean_midpoint,
    mean_relative_range,
    median_range,
    median_midpoint,
    box_range,
    frac_variable,
    frac_zero_fixed,
    frac_bidirectional,
    mean_abs_flux,
    std_range,
    median_midpoint_over_range_norm,
]


# ================================================== GLOBAL METRICS ==================================================


def _rxn_index(fva_reactions: list[str], reaction_id: str) -> int:
    try:
        return fva_reactions.index(reaction_id)
    except ValueError:
        raise ValueError(f"Reaction '{reaction_id}' not found in fva_reactions")


def _cluster_mask(grid_clusters: np.ndarray, cluster_index: int) -> np.ndarray:
    return grid_clusters == cluster_index


def _filtered_minmax(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int, reaction_id: str
    ) -> np.ndarray:
    """Returns: array shape (n_points_in_cluster, 2) with [min,max]"""
    idx = _rxn_index(fva_reactions, reaction_id)
    mask = _cluster_mask(grid_clusters, cluster_index)
    return fva_results[mask, idx, :]


def _ratio_metric(
        fva_reactions: list[str], 
        fva_results: np.ndarray, 
        grid_clusters: np.ndarray, 
        cluster_index: int, 
        reaction_tuple: tuple[str, str], 
        num_func: Callable[[Floating, Floating], Floating] | None = None, 
    ) -> float:
    rxn1_id, rxn2_id = reaction_tuple

    rxn1 = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn1_id))
    rxn2 = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn2_id))

    m1 = float(np.median(rxn1))
    m2 = float(np.median(rxn2))

    num = num_func(m1, m2) if num_func is not None else m1

    ratio = _safe_div(num, m1+m2)
    return float(ratio)


def s_rubisco_midpoint_median(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("RUBISC_h", "RUBISO_h"))


def photons_per_rubisc_midpoint_median(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    num_func = lambda x,y: np.abs(x)
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_photon_e", "RUBISC_h"), num_func=num_func)


def no3_per_rubisc_midpoint_median(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    num_func = lambda x,y: x - y
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_no3_e", "RUBISC_h"), num_func=num_func)


def co2_per_rubisc_midpoint_median(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    num_func = lambda x,y: x - y
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_co2_e", "RUBISC_h"), num_func=num_func)


def _all_reaction_ranges(fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int) -> np.ndarray:
    mask = _cluster_mask(grid_clusters, cluster_index)
    filtered = fva_results[mask, :, :]  # (n_points, n_rxns, 2)
    ranges = filtered[:, :, 1] - filtered[:, :, 0]
    return ranges


def mean_range_all_reactions(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    ranges = _all_reaction_ranges(fva_results, grid_clusters, cluster_index)
    return float(np.mean(ranges))


def std_range_all_reactions(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    ranges = _all_reaction_ranges(fva_results, grid_clusters, cluster_index)
    return float(np.std(ranges))


def blocked_fraction_all_reactions(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int, delta: float = DELTA
    ) -> float:
    ranges = _all_reaction_ranges(fva_results, grid_clusters, cluster_index)
    blocked = np.all(np.abs(ranges) < delta, axis=0)  # (n_rxns,)
    return float(np.mean(blocked))


def _median_range(
        fva_reactions, fva_results, grid_clusters, cluster_index, reaction_id
    ) -> float:
    minmax = _filtered_minmax(
        fva_reactions, fva_results, grid_clusters, cluster_index, reaction_id
    )
    return float(np.median(_range(minmax)))


def no3_to_co2_capacity_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    r_no3 = abs(_median_range(fva_reactions, fva_results, grid_clusters, cluster_index, "EX_no3_e"))
    r_co2 = abs(_median_range(fva_reactions, fva_results, grid_clusters, cluster_index, "EX_co2_e"))
    return float(_safe_div(r_no3 - r_co2, r_no3 + r_co2))


def _no3_per_N_biomass(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int,
    ) -> tuple[float, float]:
    # numerator
    no3_mid = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, "EX_no3_e"))
    no3 = float(np.abs(np.median(no3_mid)))

    # denominator
    N_rxns = ["biomass_pro_c", "biomass_DNA_c", "biomass_RNA_c"]

    mids = []
    for rxn in N_rxns:
        mid = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn))
        mids.append(np.abs(np.median(mid)))

    N_biomass = float(np.sum(mids))

    return float(_safe_div(no3 - N_biomass, no3 + N_biomass)), float(no3 + N_biomass)


def no3_per_N_biomass_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int,
    ) -> float:
    return _no3_per_N_biomass(fva_reactions, fva_results, grid_clusters, cluster_index)[0]


def no3_per_N_biomass_sum(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int,
    ) -> float:
    return _no3_per_N_biomass(fva_reactions, fva_results, grid_clusters, cluster_index)[1]


def _co2_per_C_biomass(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int,
    ) -> tuple[float, float]:
    # numerator
    co2_mid = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, "EX_co2_e"))
    co2 = float(np.abs(np.median(co2_mid)))

    # denominator
    C_rxns = ["biomass_mem_lipids_c", "biomass_carb_c", "biomass_TAG_c"]

    mids = []
    for rxn in C_rxns:
        mid = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn))
        mids.append(np.abs(np.median(mid)))

    C_biomass = float(np.sum(mids))

    return float(_safe_div(co2 - C_biomass, co2 + C_biomass)), float(co2 + C_biomass)


def co2_per_C_biomass_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int,
    ) -> float:
    return _co2_per_C_biomass(fva_reactions, fva_results, grid_clusters, cluster_index)[0]


def co2_per_C_biomass_sum(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int,
    ) -> float:
    return _co2_per_C_biomass(fva_reactions, fva_results, grid_clusters, cluster_index)[1]


GLOBAL_METRIC_LIST = [
    no3_to_co2_capacity_ratio,
    no3_per_N_biomass_ratio,
    no3_per_N_biomass_sum,
    co2_per_C_biomass_ratio,
    co2_per_C_biomass_sum,
    s_rubisco_midpoint_median,
    photons_per_rubisc_midpoint_median,
    no3_per_rubisc_midpoint_median,
    co2_per_rubisc_midpoint_median,
    mean_range_all_reactions,
    std_range_all_reactions,
    blocked_fraction_all_reactions,
]
