import logging
from typing import Callable
from functools import wraps

import numpy as np

from .constants import NON_ZERO_TOLERANCE, EPS, Floating


def _midpoint(minmax: np.ndarray) -> np.ndarray:
    return 0.5 * (minmax[:, 0] + minmax[:, 1])


def _range(minmax: np.ndarray) -> np.ndarray:
    return minmax[:, 1] - minmax[:, 0]


def _safe_div(a: Floating, b: Floating, eps: float = EPS) -> Floating:
    return a / (b + eps)


# ====================================== REACTION METRICS ======================================


def minimum(minmax: np.ndarray) -> float:
    """Minimum feasible flux across all points (lower bound)."""
    return float(np.min(minmax[:, 0]))


def maximum(minmax: np.ndarray) -> float:
    """Maximum feasible flux across all points (upper bound)."""
    return float(np.max(minmax[:, 1]))


def mean_range(minmax: np.ndarray) -> float:
    """Mean flux variability range across points."""
    r = _range(minmax)
    return float(np.mean(r))


def mean_midpoint(minmax: np.ndarray) -> float:
    """Mean midpoint of the feasible flux interval."""
    mid = _midpoint(minmax)
    return float(np.mean(mid))


def median_range(minmax: np.ndarray) -> float:
    """Median flux variability range."""
    r = _range(minmax)
    return float(np.median(r))


def median_midpoint(minmax: np.ndarray) -> float:
    """Median midpoint of the feasible flux interval."""
    mid = _midpoint(minmax)
    return float(np.median(mid))


def std_range(minmax: np.ndarray) -> float:
    """Standard deviation of flux variability ranges."""
    r = _range(minmax)
    return float(np.std(r))


def frac_variable(minmax: np.ndarray, delta: float = NON_ZERO_TOLERANCE) -> float:
    """Fraction of points with non-negligible flux variability."""
    r = _range(minmax)
    return float(np.mean(r > delta))


def frac_fixed(minmax: np.ndarray, delta: float = NON_ZERO_TOLERANCE) -> float:
    """Fraction of points with negligible flux variability"""
    r = _range(minmax)
    return float(np.mean(r < delta))


def frac_bidirectional(minmax: np.ndarray, delta: float = NON_ZERO_TOLERANCE) -> float:
    """Fraction of points allowing flux in both directions."""
    return float(np.mean((minmax[:, 0] < -delta) & (minmax[:, 1] > delta)))


def mean_abs_flux(minmax: np.ndarray) -> float:
    """Mean absolute flux capacity across points."""
    cap = np.maximum(np.abs(minmax[:, 0]), np.abs(minmax[:, 1]))
    return float(np.mean(cap))


REACTION_METRIC_LIST: list[Callable] = [
    minimum,
    maximum,
    mean_range,
    mean_midpoint,
    median_range,
    median_midpoint,
    mean_abs_flux,
    std_range,
    frac_variable,
    frac_fixed,
    frac_bidirectional,
]


# ====================================== GLOBAL METRICS ======================================


def _rxn_index(fva_reactions: list[str], reaction_id: str) -> int:
    try:
        return fva_reactions.index(reaction_id)
    except ValueError:
        raise ValueError(f"Reaction '{reaction_id}' not found in fva_reactions")


def _cluster_mask(clusters: np.ndarray, cluster_index: int) -> np.ndarray:
    return clusters == cluster_index


def _filtered_minmax(
    fva_reactions: list[str], 
    fva_results: np.ndarray, 
    clusters: np.ndarray, 
    cluster_index: int, 
    reaction_id: str,
) -> np.ndarray:
    """Returns: array shape (n_points_in_cluster, 2) with [min,max]"""
    idx = _rxn_index(fva_reactions, reaction_id)
    mask = _cluster_mask(clusters, cluster_index)
    return fva_results[mask, idx, :]


def _median_range(
    fva_reactions: list[str], 
    fva_results: np.ndarray, 
    clusters: np.ndarray, 
    cluster_index: int, 
    reaction_id: str,
) -> float:
    minmax = _filtered_minmax(fva_reactions, fva_results, clusters, cluster_index, reaction_id)
    return float(np.median(_range(minmax)))


def _all_reaction_ranges(
    fva_results: np.ndarray, clusters: np.ndarray, cluster_index: int
) -> np.ndarray:
    mask = _cluster_mask(clusters, cluster_index)
    filtered = fva_results[mask, :, :]  # (n_points, n_rxns, 2)
    ranges = filtered[:, :, 1] - filtered[:, :, 0]
    return ranges


def mean_range_all_reactions(
    fva_reactions: list[str], fva_results: np.ndarray, clusters: np.ndarray, cluster_index: int
) -> float:
    """Mean flux variability range across all reactions in the cluster."""
    ranges = _all_reaction_ranges(fva_results, clusters, cluster_index)
    return float(np.mean(ranges))


def median_range_all_reactions(
    fva_reactions: list[str], fva_results: np.ndarray, clusters: np.ndarray, cluster_index: int
) -> float:
    """Median flux variability range across all reactions in the cluster."""
    ranges = _all_reaction_ranges(fva_results, clusters, cluster_index)
    return float(np.median(ranges))


def std_range_all_reactions(
    fva_reactions: list[str], fva_results: np.ndarray, clusters: np.ndarray, cluster_index: int
) -> float:
    """Standard deviation of flux variability ranges across reactions."""
    ranges = _all_reaction_ranges(fva_results, clusters, cluster_index)
    return float(np.std(ranges))


def blocked_fraction_all_reactions(
    fva_reactions: list[str], 
    fva_results: np.ndarray, 
    clusters: np.ndarray, 
    cluster_index: int, 
    delta: float = NON_ZERO_TOLERANCE,
) -> float:
    """Fraction of reactions that are blocked across all points in the cluster."""
    ranges = _all_reaction_ranges(fva_results, clusters, cluster_index)
    blocked = np.all(np.abs(ranges) < delta, axis=0)  # (n_rxns,)
    return float(np.mean(blocked))


GLOBAL_METRIC_LIST: list[Callable] = [
    mean_range_all_reactions,
    median_range_all_reactions,
    std_range_all_reactions,
    blocked_fraction_all_reactions,
]


# ====================================== CUSTOM GLOBAL METRICS ======================================


def error_handler(function: Callable[..., float]) -> Callable[..., float]:
    """Decorator that handles exceptions raised by metrics that use reactions not found in fva_reactions"""
    @wraps(function)
    def wrapper(*args, **kwargs) -> float:
        try:
            return function(*args, **kwargs)
        except ValueError as e:
            if "Reaction" in str(e) and "not found in fva_reactions" in str(e):
                logging.warning(f"{e}: defaulting value to {0.0}")
                return 0.0
            raise
    return wrapper


def _aggregate_reactions(
    fva_reactions: list[str],
    fva_results: np.ndarray,
    clusters: np.ndarray,
    cluster_index: int,
    reactions: str | list[str],
) -> float:
    if isinstance(reactions, str):
        reactions = [reactions]

    values: list[float] = []

    for rxn_id in reactions:
        try:
            mid = _midpoint(
                _filtered_minmax(
                    fva_reactions,
                    fva_results,
                    clusters,
                    cluster_index,
                    rxn_id,
                )
            )
            values.append(float(np.abs(np.median(mid))))
        except ValueError as e:
            if "not found in fva_reactions" in str(e):
                logging.warning(f"{e}: defaulting value to {0.0}")
                values.append(0.0)
            else:
                raise

    return float(np.sum(values))


def ratio_metric(
    fva_reactions: list[str], 
    fva_results: np.ndarray, 
    clusters: np.ndarray, 
    cluster_index: int, 
    numerator: str | list[str],
    denominator: str | list[str],
    num_func: Callable[[Floating, Floating], Floating] | None = None, 
    den_func: Callable[[Floating, Floating], Floating] | None = None, 
) -> float:
    m1 = _aggregate_reactions(
        fva_reactions, fva_results, clusters, cluster_index, numerator
    )
    m2 = _aggregate_reactions(
        fva_reactions, fva_results, clusters, cluster_index, denominator
    )

    num = num_func(m1, m2) if num_func is not None else m1
    den = den_func(m1, m2) if den_func is not None else m2

    return float(_safe_div(num, den))


@error_handler
def no3_to_co2_capacity_ratio(
    fva_reactions: list[str], fva_results: np.ndarray, clusters: np.ndarray, cluster_index: int
) -> float:
    """Relative nitrate vs CO2 flux capacity based on median ranges."""
    r_no3 = abs(_median_range(fva_reactions, fva_results, clusters, cluster_index, "NO3t_e"))
    r_co2 = abs(_median_range(fva_reactions, fva_results, clusters, cluster_index, "CO2t_e"))
    return float(_safe_div(r_no3 - r_co2, r_no3 + r_co2))


#GLOBAL_METRIC_LIST.append(no3_to_co2_capacity_ratio)

