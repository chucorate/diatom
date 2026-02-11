from typing import Callable

import numpy as np
import pandas as pd

import prince
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score

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
    ) -> tuple[float, float]:
    rxn1_id, rxn2_id = reaction_tuple

    rxn1 = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn1_id))
    rxn2 = _midpoint(_filtered_minmax(fva_reactions, fva_results, grid_clusters, cluster_index, rxn2_id))

    m1 = float(np.median(np.abs(rxn1)))
    m2 = float(np.median(np.abs(rxn2)))

    num = num_func(m1, m2) if num_func is not None else m1

    simple_ratio = _safe_div(num, m2)
    ratio = _safe_div(num, m1 + m2)
    return float(ratio), float(simple_ratio)


def s_rubisco(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("RUBISC_h", "RUBISO_h"))[0]


def photons_per_rubisc_difference_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    num_func = lambda x,y: x - y
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_photon_e", "RUBISC_h"), num_func=num_func)[0]


def photons_per_rubisc_simple_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_photon_e", "RUBISC_h"))[1]


def no3_per_rubisc_difference_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    num_func = lambda x,y: x - y
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_no3_e", "RUBISC_h"), num_func=num_func)[0]


def no3_per_rubisc_simple_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_no3_e", "RUBISC_h"))[1]


def co2_per_rubisc_difference_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    num_func = lambda x,y: x - y
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_co2_e", "RUBISC_h"), num_func=num_func)[0]


def co2_per_rubisc_simple_ratio(
        fva_reactions: list[str], fva_results: np.ndarray, grid_clusters: np.ndarray, cluster_index: int
    ) -> float:
    return _ratio_metric(fva_reactions, fva_results, grid_clusters, cluster_index, ("EX_co2_e", "RUBISC_h"))[1]


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

    return float(_safe_div(no3, no3 + N_biomass)), float(no3 + N_biomass)


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

    return float(_safe_div(co2, co2 + C_biomass)), float(co2 + C_biomass)


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
    s_rubisco,
    photons_per_rubisc_simple_ratio,
    no3_per_rubisc_simple_ratio,
    co2_per_rubisc_simple_ratio,
    photons_per_rubisc_difference_ratio,
    no3_per_rubisc_difference_ratio,
    co2_per_rubisc_difference_ratio,
    mean_range_all_reactions,
    std_range_all_reactions,
    blocked_fraction_all_reactions,
]


# ================================================== FEATURE SELECTION SCORE METRICS ==================================================


def first_mixed_merge_height(reaction_states: np.ndarray, linkage_matrix: np.ndarray, **kwargs) -> float:
    """Returns the dendrogram height at which points with different qualitative states for a given reaction are first merged.

    Parameters
    ----------
    linkage_matrix
        Output of scipy.cluster.hierarchy.linkage, shape (n-1, 4)
    reaction_states
        Qualitative states of a single reaction across grid points,
        shape (n_points,)

    Returns
    -------
    float
        Merge height. Larger means earlier (more explanatory).
    """
    n_points = reaction_states.shape[0]
    reaction_states = np.round(reaction_states, 2)
    
    # cluster_id -> set of point indices
    active_clusters: dict[int, list[int]] = {i: [i] for i in range(n_points)}

    for merge_index, (left, right, height, _) in enumerate(linkage_matrix):
        left_id = int(left)
        right_id = int(right)
        new_cluster_id = n_points + merge_index

        merged_points = active_clusters[left_id] + active_clusters[right_id]
        active_clusters[new_cluster_id] = merged_points

        del active_clusters[left_id], active_clusters[right_id]

        merged_states = np.unique(reaction_states[merged_points])

        if len(merged_states) > 1:
            return float(height)

    return -1.0 # reaction never separates clusters


def intra_inter(reaction_states: np.ndarray, clusters: np.ndarray, **kwargs) -> float:
    cluster_ids = np.unique(clusters)

    # ---------- intra ----------
    intra_values: list[float] = []
    for c in cluster_ids:
        cluster_states = reaction_states[clusters == c]
        purity = _cluster_purity(cluster_states)
        intra_values.append(1.0 - purity)

    D_intra = float(np.mean(intra_values))

    # ---------- inter ----------
    inter_values: list[float] = []
    for i, c1 in enumerate(cluster_ids):
        states_1 = reaction_states[clusters == c1]
        for c2 in cluster_ids[i + 1:]:
            states_2 = reaction_states[clusters == c2]
            inter_values.append(
                _inter_cluster_disagreement(states_1, states_2)
            )

    D_inter = float(np.mean(inter_values)) if inter_values else 0.0

    return D_inter - D_intra


def _cluster_purity(states: np.ndarray) -> float:
    """Computes purity of a categorical vector."""
    values, counts = np.unique(states, return_counts=True)
    return counts.max() / counts.sum()


def _inter_cluster_disagreement(states_a: np.ndarray, states_b: np.ndarray) -> float:
    """Computes categorical disagreement rate between two clusters.

    Parameters
    ----------
    states_a, states_b
        Qualitative states of a reaction in two different clusters.

    Returns
    -------
    float
        Disagreement rate in [0, 1].
    """
    values_a, counts_a = np.unique(states_a, return_counts=True)
    values_b, counts_b = np.unique(states_b, return_counts=True)

    total_pairs = counts_a.sum() * counts_b.sum()
    same_pairs = 0

    freq_a = dict(zip(values_a, counts_a))
    freq_b = dict(zip(values_b, counts_b))

    for v in freq_a.keys() & freq_b.keys():
        same_pairs += freq_a[v] * freq_b[v]

    return 1.0 - same_pairs / total_pairs


def mutual_information(reaction_states: np.ndarray, clusters: np.ndarray, **kwargs) -> float:
    return float(mutual_info_score(reaction_states, clusters))


def entropy(reaction_states: np.ndarray, eps: float = 1e-12, **kwargs) -> float:
    """Entropía categórica H(X)."""
    values, counts = np.unique(reaction_states, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + eps))


def std_normalized(fva_result: np.ndarray, eps: float = 1e-12, **kwargs) -> float:
    ranges = fva_result[:, 1] - fva_result[:, 0]  # rango por punto
    return float(np.std(ranges) / (np.mean(ranges) + eps))


PER_REACTION_SCORE_FUNCTIONS = [
    #first_mixed_merge_height,
    intra_inter,
    mutual_information,
    entropy,
    std_normalized,
    ]


def rf_importance(qual_vector_df: pd.DataFrame, grid_clusters: np.ndarray, n_estimators: int = 100, **kwargs) -> pd.Series:
    """Identifica qué reacciones son las más importantes para predecir la pertenencia a los clusters usando Random Forest."""
    if qual_vector_df.empty or grid_clusters.size == 0:
        raise RuntimeError("Datos insuficientes (DF vacío o clusters no calculados).")

    X = qual_vector_df.values
    y = grid_clusters

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_importance_series = pd.Series(importances, index=qual_vector_df.columns).sort_values(ascending=False)

    return feature_importance_series


def mca_score(qual_vector_df: pd.DataFrame, n_components: int = 5, state_prefix: str = "s", min_nunique: int = 2, random_state: int = 42, **kwargs) -> pd.Series:
    """MCA no supervisado para selección de reacciones a partir de qual_vector_df.

    Parameters
    ----------
    qual_vector_df : pd.DataFrame
        Filas = observaciones, columnas = reacciones,
        valores = estados cualitativos (numéricos pero categóricos).
    n_components : int
        Número de ejes MCA.
    state_prefix : str
        Prefijo para codificar estados (evita ambigüedad numérica).
    min_nunique : int
        Número mínimo de estados distintos para conservar una reacción.
    random_state : int

    Returns
    -------
    pd.Series
        Índice = reacción, valor = score MCA (mayor = más explicativa).
    """
    # filter non informative columns
    informative_cols = qual_vector_df.columns[qual_vector_df.nunique(dropna=True) >= min_nunique]
    if len(informative_cols) == 0:
        raise ValueError("No hay reacciones con variabilidad suficiente para MCA.")
    Q = qual_vector_df[informative_cols].copy()

    # force categorical data type
    X_cat = Q.fillna("NaN").astype(int, errors="ignore").astype(str)
    X_cat = X_cat.apply(lambda col: state_prefix + col)

    mca = prince.MCA(n_components=n_components, n_iter=10, random_state=random_state, ).fit(X_cat)

    # index format: "REACTION__sSTATE"
    cat_contrib = (mca.column_contributions_.iloc[:, :n_components].fillna(0.0))

    reaction_scores = (cat_contrib.groupby(lambda s: s.split("__")[0]).sum().sum(axis=1).sort_values(ascending=False))
    return reaction_scores


GLOBAL_SCORE_FUNCTIONS = [
    rf_importance,
    mca_score,
]

