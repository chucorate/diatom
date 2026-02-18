import numpy as np
import pandas as pd
import prince
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score


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
    """Difference between inter-cluster disagreement and intra-cluster heterogeneity."""
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
    """Mutual information between reaction states and cluster assignments."""
    return float(mutual_info_score(reaction_states, clusters))


def entropy(reaction_states: np.ndarray, eps: float = 1e-12, **kwargs) -> float:
    """Categorical entropy of reaction states."""
    values, counts = np.unique(reaction_states, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + eps))


def std_normalized(fva_result: np.ndarray, eps: float = 1e-12, **kwargs) -> float:
    """Standard deviation of flux ranges normalized by their mean."""
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
    """Reaction importance based on Random Forest prediction of cluster labels."""
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
    """Unsupervised reaction relevance score using Multiple Correspondence Analysis."""
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

