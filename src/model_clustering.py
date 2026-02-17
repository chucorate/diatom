from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy

from src.metrics import REACTION_METRIC_LIST, GLOBAL_METRIC_LIST, PER_REACTION_SCORE_FUNCTIONS, GLOBAL_SCORE_FUNCTIONS

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem
    from diatom.diatom import Diatom


class ModelClustering():
    """
    Class for managing clustering grid points and reactions based on qualitative FVA profiles,
    and for computing cluster-level summaries, metrics, and reaction scores.
    
    Parameters
    ----------
    diatom : Diatom
        Parent diatom object providing access to the metabolic model,
        grid sampler, and I/O utilities.

    Attributes
    ----------
    initial_n_clusters : int
        Number of initial clusters specified by user. It can be higher than
        actual amount of clusters produced.
    grid_n_clusters : int
        Number of actual clusters produced.
    grid_clusters : np.ndarray, shape (n_points, )
        Array containing the cluster labels of all grid points. 
    linkage_matrix : np.ndarray, shape (n_points-1, 4)
        Linkage matrix encoding the dendrogram produced via hierarchical clustering.

    """
    def __init__(self, modelclass: "Diatom"):
        self.modelclass = modelclass
        self.initial_n_clusters: int = 0
        self.grid_n_clusters: int = 0
        self.grid_clusters: np.ndarray = np.array([])
        self.linkage_matrix: np.ndarray = np.array([])


    @property
    def qual_vector(self) -> pd.DataFrame:
        return self.modelclass.analyze.qual_vector


    def one_hot_encode_reactions(self, changing: bool = True) -> np.ndarray:
        """One hot encodes qualitative states. 
        
        Optionally restrict qualitative vectors to reactions whose qualitative
        state changes across grid points.

        Parameters
        ----------
        changing : bool
            If True, keep only reactions with non-constant qualitative values.

        Returns
        -------
        encoded_reactions : np.ndarray
            One-hot encoded qualitative matrix (grid x features).
        """
        z = self.qual_vector.copy()

        if changing:
            changed_rxns = self.qual_vector.max(axis=0) != self.qual_vector.min(axis=0)
            changed_rxns_ids = z.columns[changed_rxns]
            z = z[changed_rxns_ids]

        #print(f"changing reaction: {self.qual_vector_df.values.shape} -> {changed_rxns.sum()}")
        z_one_hot = pd.get_dummies(z.astype(str))
        print(f"base: {z.shape} -> one-hot: {z_one_hot.shape}")
        return z_one_hot.values
        

    def set_grid_clusters(
        self, 
        method: str, 
        changing: bool = True, 
        initial_n_clusters: int = 20, 
        **kwargs
    ) -> None:
        """Cluster grid points based on qualitative flux vectors.

        Uses pairwise Jaccard distances between grid points and stores the resulting
        cluster labels and number of clusters as attributes.

        Parameters
        ----------
        method : str
            Clustering method identifier (currently only 'hierarchical').
        changing : bool
            If True, restrict to reactions that change across the grid.
        initial_n_clusters : int
            Initial target number of clusters .
   
        Attributes Set
        --------------
        initial_n_clusters : int 
            Argument is passed to the clustering backend.
        grid_n_clusters : int
            Number of clusters produced.
        grid_clusters : np.ndarray, shape (n_points, )
            Array containing the cluster labels of all grid points. 
        linkage_matrix : np.ndarray, shape (n_points-1, 4)
            Linkage matrix encoding the dendrogram produced via hierarchical clustering.
        """
        self.modelclass._require(qual_vector=True)

        self.initial_n_clusters = initial_n_clusters
        
        loaded_clusters = self.modelclass.io.load_clusters()
        if isinstance(loaded_clusters, tuple):
            self.grid_n_clusters, self.grid_clusters = loaded_clusters
            return 

        qualitative_vector = self.one_hot_encode_reactions(changing)
        
        print("Clustering grid points ...") 
        self.grid_n_clusters, self.grid_clusters, self.linkage_matrix = self._map_clusters(method, qualitative_vector, **kwargs)
        self.modelclass.io.save_clusters(self.grid_n_clusters, self.grid_clusters)


    def _map_clusters(self, method: str, qualitative_vector: np.ndarray, **kwargs) -> tuple[int, np.ndarray, np.ndarray]:
        """Computed pairwise Jaccard distances and applies a clustering method.

        Returns
        -------
        n_clusters : int
            Number of clusters computed.
        clusters : np.ndarray, shape (n_points, )
            Cluster labels.
        linkage_matrix : np.ndarray, shape (n_points-1, 4)
            Hierarchical linkage matrix.
        """
        distance_metric = 'jaccard'
        dvector = distance.pdist(qualitative_vector, distance_metric) 
        #dmatrix = distance.squareform(dvector)     

        if method == 'hierarchical':
            n_clusters, clusters, linkage_matrix = _get_hierarchical_clusters(dvector,**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Done! n_clusters: {n_clusters}")    
        return n_clusters, clusters, linkage_matrix


    @staticmethod
    def _get_representative_qualitative_values(cluster_column: pd.Series, threshold: float) -> int | None:
        """Return the dominant qualitative value in a cluster column.
        
        A value is considered representative if it appears in at least `threshold` fraction of 
        the grid points in the cluster. If the threshold is not met, returns None."""
        total = len(cluster_column)

        qualitative_values, counts = np.unique(cluster_column, return_counts=True)
        representative = qualitative_values[counts/total >= threshold]
         
        # qualitative value present if at least threshold of reactions in cluster  
        return representative[0] if representative.size > 0 else None
         

    def get_grid_cluster_qual_profiles(
        self, 
        threshold: float = 0.75,
        changing: bool = True, 
        convert: bool = True,
        selected_reactions: list[str] | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Compute representative qualitative reaction profiles for each grid cluster.

        For each grid cluster, assigns a qualitative value to each reaction if it
        appears in at least a given fraction of grid points. Optionally filters
        reactions that change between clusters and converts qualitative codes.
        
        Parameters
        ----------
        threshold : float, default=0.75
            Minimum fraction of grid points within a cluster that must share the
            same qualitative value to be considered representative.
        changing : bool, default=True
            If True, only reactions whose representative values differ across clusters are retained.
        convert : bool, default=True
            If True, integer qualitative codes are mapped using `analyze.category_dict`.
        selected_reactions : list[str] | None, default=None
            Optional list of reaction IDs to subset the result.
        overwrite : bool, default=False
            Whether to overwrite an existing saved dataframe.

        Returns
        -------
        representatives : pd.DataFrame
            Rows correspond to reactions and columns to clusters (``c1, c2, ...``).
            Entries are representative qualitative values or NaN.
        """
        self.modelclass._require(qual_vector=True, clusters=True)
        
        vector_df = self.qual_vector.astype('int32')

        cluster_ids = np.arange(1, self.grid_n_clusters + 1)
        #print(f"cluster_ids: {cluster_ids}, grid_clusters: {self.grid_clusters}")
        cluster_dfs = [vector_df[self.grid_clusters == cluster_id] for cluster_id in cluster_ids]
        print(f"cluster_dfs len: {len(cluster_dfs)}")
        representatives_list = [
            cluster_df.apply(
                self._get_representative_qualitative_values,
                threshold=threshold,
            ) 
            for cluster_df in cluster_dfs
        ]
        
        representatives = pd.concat(representatives_list, axis=1).astype('float')
        representatives.columns = [f'c{cluster_id}' for cluster_id in cluster_ids]

        analyze = self.modelclass.analyze

        if changing: # report only reactions that have different qualitative values in at least two clusters
            changing_filter = representatives.apply(lambda x: x.unique().size > 1, axis = 1)    
            representatives = representatives[changing_filter]
        
        if convert:
            representatives = representatives.replace(analyze.category_dict)

        reaction_len = len(selected_reactions) if selected_reactions is not None else -1
        if selected_reactions is not None:
            representatives = representatives.loc[selected_reactions]
       
        self.modelclass.io.save_cluster_df(
            representatives, 
            "Qualitative_profiles", 
            reaction_len=reaction_len, 
            index=True, 
            overwrite=overwrite,
        )
        return representatives


    @staticmethod
    def compare_clusters(clusters_df: pd.DataFrame, cluster_id1: str | int, cluster_id2: str | int) -> pd.DataFrame:
        """Compare qualitative values between two clusters.
        
        Returns a dataframe whose rows only display qualitative values that are different between 
        the clusters."""
    
        if isinstance(cluster_id1, int):
            cluster_id1 = 'c%d' % cluster_id1
        if isinstance(cluster_id2, int):
            cluster_id2 = 'c%d' % cluster_id2            
        
        comparative_df = clusters_df[[cluster_id1, cluster_id2]]
        
        # filter out rows where the two clusters share values
        changing_filter = comparative_df[cluster_id1] != comparative_df[cluster_id2]
        comparative_df = comparative_df[changing_filter]

        return comparative_df
    

    def get_cluster_global_metrics(self, reaction_list: list[str], overwrite: bool = False) -> pd.DataFrame:
        """Compute global metrics for each grid cluster.

        Each metric in `GLOBAL_METRIC_LIST` is evaluated independently on every
        cluster, producing a single scalar value per (cluster, metric) pair.

        Parameters
        ----------
        reaction_list : list[str]
            List of reaction identifiers used only for bookkeeping when saving results to disk. 
            The values themselves are not used in the metric computation.
        overwrite : bool, default=False
            Whether to overwrite an existing saved dataframe on disk.

        Returns
        -------
        df : pd.DataFrame
            Long-form dataframe with columns:
            - `cluster` : int  
            Cluster identifier.
            - `metric` : str  
            Name of the global metric.
            - `value` : float
            Metric value for the given cluster.
        """
        self.modelclass._require(clusters=True)

        grid_clusters = self.grid_clusters
        fva_reactions = self.modelclass.analyze.fva_reactions
        fva_results = self.modelclass.analyze.fva_results

        metric_names = [metric.__name__ for metric in GLOBAL_METRIC_LIST]

        rows: list[dict[str, Any]] = []
        for cluster_index in range(1, self.grid_n_clusters+1):
            metric_results = [metric(fva_reactions, fva_results, grid_clusters, cluster_index) for metric in GLOBAL_METRIC_LIST]

            for metric_name, metric_value in zip(metric_names, metric_results):
                rows.append({
                    "cluster": cluster_index,
                    "metric": metric_name,
                    "value": metric_value
                })

        df = pd.DataFrame(rows)
        self.modelclass.io.save_cluster_df(df, "Global_metrics", reaction_len=len(reaction_list), metric_list=GLOBAL_METRIC_LIST, overwrite=overwrite)
        return df
    

    def get_cluster_metrics_per_reaction(self, reaction_list: list[str], overwrite: bool = False) -> pd.DataFrame:
        """Compute per-reaction metrics for each grid cluster.

        For every reaction in `reaction_list` and every cluster, metrics defined
        in `REACTION_METRIC_LIST` are computed using the FVA results restricted
        to that reaction and cluster.

        Parameters
        ----------
        reaction_list : list[str]
            List of reaction identifiers for which metrics are computed.
        overwrite : bool, default=False
            Whether to overwrite an existing saved dataframe on disk.

        Returns
        -------
        df : pd.DataFrame
            Long-form dataframe with columns:
            - `reaction_id` : str  
            Reaction identifier.
            - `cluster` : int  
            Cluster identifier.
            - `metric` : str  
            Name of the reaction-level metric.
            - `value` : float
            Metric value for the given reaction and cluster.
        """
        self.modelclass._require(clusters=True)

        grid_clusters = self.grid_clusters
        fva_reactions = self.modelclass.analyze.fva_reactions
        fva_results = self.modelclass.analyze.fva_results

        metric_names = [metric.__name__ for metric in REACTION_METRIC_LIST]

        rows: list[dict[str, Any]] = []
        for reaction_id in reaction_list:
            reaction_index = fva_reactions.index(reaction_id)
            reaction_fva_results = (fva_results[:, reaction_index, :])

            for cluster_index in range(1, self.grid_n_clusters+1):
                filtered_results = reaction_fva_results[grid_clusters == cluster_index]
                metric_results = [metric(filtered_results) for metric in REACTION_METRIC_LIST]

                for metric_name, metric_value in zip(metric_names, metric_results):
                    rows.append({
                        "reaction_id": reaction_id,
                        "cluster": cluster_index,
                        "metric": metric_name,
                        "value": metric_value
                    })

        df = pd.DataFrame(rows)
        self.modelclass.io.save_cluster_df(df, "Metrics_per_reaction", reaction_len=len(reaction_list), metric_list=REACTION_METRIC_LIST, overwrite=overwrite)
        return df

    
    def reaction_scores(self, sort_score: bool = True, sort_index: int = 0, **kwargs) -> tuple[pd.DataFrame, list[str]]:
        """Compute reaction-level scores and perform consensus feature selection.

        This method evaluates multiple scoring functions at the reaction level,
        combining global (cluster-wise) and per-reaction criteria. The resulting
        score matrix is then used to select a subset of reactions via consensus
        voting.

        Returns
        -------
        score_df : pd.DataFrame
            DataFrame indexed by reaction ID. Columns correspond to score functions from 
            `GLOBAL_SCORE_FUNCTIONS` and `PER_REACTION_SCORE_FUNCTIONS`.
        selected_reactions : list[str]
            Reactions selected via consensus voting.

        Notes
        -----
        - Global score functions operate on the full qualitative vector.
        - Per-reaction score functions additionally depend on the linkage matrix
        and reaction-specific FVA results.
        """
        self.modelclass._require(clusters=True, qual_vector=True)

        df = pd.DataFrame(index=self.qual_vector.columns)

        for score_func in GLOBAL_SCORE_FUNCTIONS:
            df[score_func.__name__] = score_func(qual_vector_df=self.qual_vector, grid_clusters=self.grid_clusters)

        for score_func in PER_REACTION_SCORE_FUNCTIONS:
            func_name = score_func.__name__
            scores = []
            for rid in self.qual_vector.columns:
                fva_reactions = self.modelclass.analyze.fva_reactions
                fva_results = self.modelclass.analyze.fva_results
                reaction_index = fva_reactions.index(rid)
                val = score_func(
                    reaction_states=self.qual_vector[rid].values,
                    clusters=self.grid_clusters,
                    linkage_matrix=self.linkage_matrix,
                    fva_result = (fva_results[:, reaction_index, :])
                )
                scores.append(val)
            df[func_name] = scores

        col_to_sort = df.columns[sort_index] if sort_index < len(df.columns) else df.columns[0]
        if sort_score:
            df = df.sort_values(by=col_to_sort, ascending=False)

        metric_names = [f.__name__ for f in GLOBAL_SCORE_FUNCTIONS] + [f.__name__ for f in PER_REACTION_SCORE_FUNCTIONS]
        feature_selection = _consensus_feature_selection(df, metric_names, **kwargs)

        return df, feature_selection
    

# ======================================================= CLUSTER FUNCTIONS =======================================================


def _get_hierarchical_clusters(
    dvector: np.ndarray, 
    k: int = 20, 
    lmethod: str = 'complete', 
    criterion: str= 'maxclust', 
    **kwargs
) -> tuple[int, np.ndarray, np.ndarray]:
    """Performs hierarchical clustering."""
    linkage_matrix = hierarchy.linkage(dvector, method=lmethod)
    clusters = fcluster(linkage_matrix, t=k, criterion=criterion)

    k = len(np.unique(clusters))
    return k, clusters, linkage_matrix # clusters are indexed from 1


def _consensus_feature_selection(
    score_df: pd.DataFrame, 
    score_cols: list[str], 
    k: int = 30, 
    min_votes: int | None = 2, 
    normalize: bool = True, 
    show: bool = True,
    **kwargs,
) -> list[str]:
    """
    Consensus feature selection via top-k voting across multiple scores.

    Parameters
    ----------
    df : DataFrame
        Index = reaction_id, columns = scores
    score_cols : list of str
        Columns to use as scores
    k : int
        Top-k reactions per score
    min_votes : int or None
        Minimum number of appearances required. If None, uses majority rule.
    normalize : bool
        Rank-based normalization per score

    Returns
    -------
    DataFrame sorted by votes and mean rank
    """
    ranks = {}

    for col in score_cols:
        s = score_df[col].copy()
        r = s.rank(ascending=False, method="average") if normalize else s
        ranks[col] = r

    rank_df = pd.DataFrame(ranks)
    mean_rank = rank_df.mean(axis=1)

    # top-k mask per score
    topk_mask = rank_df <= k
    votes = topk_mask.sum(axis=1)

    if min_votes is None:
        min_votes = int(np.ceil(len(score_cols) / 2))

    out = pd.DataFrame({"votes": votes, "mean_rank": mean_rank})
    out = out[votes >= min_votes]
    out = out.sort_values(["votes", "mean_rank"], ascending=[False, True]) # type: ignore

    if show:
        print(out.to_string())

    reaction_selection = list(out.index)
    return reaction_selection

