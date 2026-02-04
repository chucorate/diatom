from typing import Callable

import numpy as np

DELTA = 1e-6
EPS = 1e-9


def _midpoint(minmax: np.ndarray) -> np.ndarray:
    return 0.5 * (minmax[:, 0] + minmax[:, 1])


def _range(minmax: np.ndarray) -> np.ndarray:
    return minmax[:, 1] - minmax[:, 0]


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
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


def std(minmax: np.ndarray) -> float:
    r = _range(minmax)
    return float(np.std(r))


REACTION_METRIC_LIST: list[Callable] = [
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
    std,
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
        num_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, 
        den_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> float:
    rxn1_id, rxn2_id = reaction_tuple

    rxn1 = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn1_id))
    rxn2 = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn2_id))

    num = num_func(rxn1, rxn2) if num_func is not None else rxn1
    den = den_func(rxn1, rxn2) if den_func is not None else rxn2

    ratio = _safe_div(num, den)
    return float(np.median(ratio))


def s_rubisco_midpoint_median(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    den_func = lambda x,y: x + y
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("RUBISC_h", "RUBISO_h"), den_func=den_func)


def rubisc_to_rubiso_midpoint_ratio_median(
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
    num_func = lambda x,y: np.abs(x)
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_no3_e", "RUBISC_h"), num_func=num_func)


def co2_per_rubisc_midpoint_median(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    num_func = lambda x,y: np.abs(x)
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


GLOBAL_METRIC_LIST = [
    s_rubisco_midpoint_median,
    rubisc_to_rubiso_midpoint_ratio_median,
    photons_per_rubisc_midpoint_median,
    no3_per_rubisc_midpoint_median,
    co2_per_rubisc_midpoint_median,
    mean_range_all_reactions,
    std_range_all_reactions,
    blocked_fraction_all_reactions,
]
