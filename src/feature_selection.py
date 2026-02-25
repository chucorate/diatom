import numpy as np
import pandas as pd
import prince
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score

from .constants import EPS


# Unsupervised scores


def entropy(reaction_states: np.ndarray, eps: float = EPS, **kwargs) -> float:
    """Categorical entropy of reaction states."""
    values, counts = np.unique(reaction_states, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + eps))


def coefficient_of_variation(fva_result: np.ndarray, eps: float = EPS, **kwargs) -> float:
    """Standard deviation of flux ranges normalized by their mean."""
    ranges = np.abs(fva_result[:, 1] - fva_result[:, 0]) 
    return float(np.std(ranges) / (np.mean(ranges) + eps))


def mca_score(
    qualitative_matrix: pd.DataFrame, 
    n_components: int = 5, 
    state_prefix: str = "s", 
    min_nunique: int = 2, 
    random_state: int = 42, 
    **kwargs
) -> pd.Series:
    """Unsupervised reaction relevance score using Multiple Correspondence Analysis."""
    informative_cols = qualitative_matrix.columns[
        qualitative_matrix.nunique(dropna=True) >= min_nunique
    ]
    if len(informative_cols) == 0:
        raise ValueError("No hay reacciones con variabilidad suficiente para MCA.")
    Q = qualitative_matrix[informative_cols].copy()

    # force categorical data type
    X_cat = Q.fillna("NaN").astype(int, errors="ignore").astype(str)
    X_cat = X_cat.apply(lambda col: state_prefix + col)

    mca = prince.MCA(n_components=n_components, n_iter=10, random_state=random_state, ).fit(X_cat)

    # index format: "REACTION__sSTATE"
    cat_contrib = (mca.column_contributions_.iloc[:, :n_components].fillna(0.0))

    reaction_scores = (
        cat_contrib.groupby(lambda s: s.split("__")[0]).sum().sum(axis=1).sort_values(ascending=False)
    )
    return reaction_scores


# Supervised scores


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


def rf_importance(
    qualitative_matrix: pd.DataFrame, clusters: np.ndarray, n_estimators: int = 100, **kwargs
) -> pd.Series:
    """Reaction importance based on Random Forest prediction of cluster labels."""
    if qualitative_matrix.empty or clusters.size == 0:
        raise RuntimeError("Datos insuficientes (DF vacío o clusters no calculados).")

    X = qualitative_matrix.values
    y = clusters

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_importance_series = pd.Series(
        importances, index=qualitative_matrix.columns
    ).sort_values(ascending=False)

    return feature_importance_series


GLOBAL_SCORE_FUNCTIONS = [
    rf_importance,
    mca_score,
]


PER_REACTION_SCORE_FUNCTIONS = [
    intra_inter,
    mutual_information,
    entropy,
    coefficient_of_variation,
    ]
