import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy

from src.metrics import REACTION_METRIC_LIST, GLOBAL_METRIC_LIST
from src.feature_selection import PER_REACTION_SCORE_FUNCTIONS, GLOBAL_SCORE_FUNCTIONS

if TYPE_CHECKING:
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
        self.grid_n_clusters: int 
        self.grid_clusters: np.ndarray 
        self.linkage_matrix: np.ndarray 
        self.representatives: pd.DataFrame


    @property
    def qualitative_matrix(self) -> pd.DataFrame:
        return self.modelclass.analyze.qualitative_matrix


    def one_hot_encode_reactions(self, changing: bool) -> np.ndarray:
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
        z = cast(pd.DataFrame, self.qualitative_matrix.copy())

        if changing:
            changed_rxns = self.qualitative_matrix.max(axis=0) != self.qualitative_matrix.min(axis=0)
            changed_rxns_ids = z.columns[changed_rxns]
            z = z[changed_rxns_ids]

        z_one_hot = pd.get_dummies(z.astype(str))
        logging.debug(f"base: {z.shape} -> one-hot: {z_one_hot.shape}")
        return z_one_hot.values
        

    def set_grid_clusters(self, n_clusters: int, method: str, changing: bool = True, **kwargs) -> None:
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
        self.modelclass._require(qualitative_matrix=True)
        
        self.initial_n_clusters = n_clusters
        
        """
        loaded_clusters = self.modelclass.io.load_clusters()
        if isinstance(loaded_clusters, tuple):
            self.grid_n_clusters, self.grid_clusters = loaded_clusters
            return 
        """

        qualitative_vector = self.one_hot_encode_reactions(changing)
        
        logging.info("Clustering grid points ...") 
        self.grid_n_clusters, self.grid_clusters, self.linkage_matrix = (
            self._map_clusters(qualitative_vector, n_clusters, method, **kwargs)
        )
        self.modelclass.io.save_clusters(self.grid_n_clusters, self.grid_clusters)


    @staticmethod
    def _map_clusters(
        qualitative_vector: np.ndarray, 
        n_clusters: int, 
        method: str, 
        criterion: str = 'maxclust', 
        metric: str = 'jaccard', 
        **kwargs,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Computed pairwise Jaccard distances and performs hierarchical clustering.

        Returns
        -------
        n_clusters : int
            Number of clusters computed.
        clusters : np.ndarray, shape (n_points, )
            Cluster labels.
        linkage_matrix : np.ndarray, shape (n_points-1, 4)
            Hierarchical linkage matrix.
        """
        dvector = distance.pdist(qualitative_vector, metric) # type: ignore

        linkage_matrix = hierarchy.linkage(dvector, method=method)
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion=criterion, **kwargs) # clusters are indexed from 1

        n_clusters = len(np.unique(clusters))
        
        logging.info(f"Done! Obtained {n_clusters} from hierarchical clustering.")
        logging.debug(f"method: {method}, criterion: {criterion}, metric: {metric}.")    
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
        self.modelclass._require(qualitative_matrix=True, clusters=True)
        
        vector_df = self.qualitative_matrix.astype('int32')

        cluster_ids = np.arange(1, self.grid_n_clusters + 1)
        cluster_dfs = [vector_df[self.grid_clusters == cluster_id] for cluster_id in cluster_ids]
        logging.debug(f"cluster_dfs len: {len(cluster_dfs)}")

        representatives_list = [
            cluster_df.apply(
                self._get_representative_qualitative_values,
                threshold=threshold,
            ) 
            for cluster_df in cluster_dfs
        ]
        
        representatives = cast(pd.DataFrame, pd.concat(representatives_list, axis=1).astype('float'))
        representatives.columns = [f'c{cluster_id}' for cluster_id in cluster_ids]

        analyze = self.modelclass.analyze

        if changing:
            changing_filter = representatives.apply(lambda x: x.unique().size > 1, axis = 1)    
            representatives = representatives[changing_filter]
        
        if selected_reactions:
            representatives = representatives.loc[selected_reactions] 

        representatives = representatives.replace(analyze.category_dict)
       
        self.modelclass.io.save_cluster_df(
            representatives, 
            "Qualitative_profiles", 
            reaction_len=len(selected_reactions) if selected_reactions is not None else -1, 
            index=True, 
            overwrite=overwrite,
        )
        self.representatives = representatives
        return representatives
    

    def n_clusters_score(self, threshold: float) -> tuple[float, pd.Series]:
        """Computes agreement scores between qualitative states and
        cluster representative profiles for each reaction.

        This method counts the fraction of grid points whose qualitative state matches 
        the representative qualitative value of their assigned cluster. 

        A reaction is considered to be successfully represented if its score is greater 
        than or equal  to `threshold`.

        Parameters
        ----------
        threshold : float
            Minimum agreement fraction required for a reaction to be considered
            successful.

        Returns
        -------
        success_ratio : float
            Fraction of reactions whose agreement score is greater than or equal
            to `threshold`.
        scores : pd.Series
            Reaction-wise agreement scores indexed by reaction ID.
        """
        self.modelclass._require(clusters=True, qualitative_matrix=True)

        reaction_list = self.representatives.index.tolist()

        n_points = self.qualitative_matrix.shape[0]
        qual_states = self.qualitative_matrix[reaction_list].replace(self.modelclass.analyze.category_dict)
        
        n_clusters = self.grid_n_clusters
        cluster_masks = {cluster_id: self.grid_clusters == cluster_id for cluster_id in range(1, n_clusters+1)}
    
        scores_dict: dict[str, float] = {}
        for reaction_id in reaction_list:
            reaction_score = 0
            representatives = self.representatives.loc[reaction_id]

            for cluster_id, cluster_mask in cluster_masks.items():
                representative_state = str(representatives[f"c{cluster_id}"])
                actual_states = qual_states.loc[cluster_mask, reaction_id]

                reaction_score += float(np.sum(actual_states == representative_state))
            
            scores_dict[reaction_id] = reaction_score / n_points

        scores = pd.Series(scores_dict)
        success_ratio = float((scores >= threshold).mean())

        return success_ratio, scores


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
            metric_results = [
                metric(fva_reactions, fva_results, grid_clusters, cluster_index) 
                for metric in GLOBAL_METRIC_LIST
            ]

            for metric_name, metric_value in zip(metric_names, metric_results):
                rows.append({
                    "cluster": cluster_index, "metric": metric_name, "value": metric_value
                })

        df = pd.DataFrame(rows)
        self.modelclass.io.save_cluster_df(
            df, 
            "Global_metrics", 
            reaction_len=len(reaction_list), 
            metric_list=GLOBAL_METRIC_LIST, 
            overwrite=overwrite,
        )
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
        self.modelclass.io.save_cluster_df(
            df, 
            "Metrics_per_reaction", 
            reaction_len=len(reaction_list), 
            metric_list=REACTION_METRIC_LIST, 
            overwrite=overwrite,
        )
        return df

    
    def reaction_scores(
        self, sort_score: bool = True, sort_index: int = 0, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
        self.modelclass._require(clusters=True, qualitative_matrix=True)

        scores_df = pd.DataFrame(index=self.qualitative_matrix.columns)

        for score_func in GLOBAL_SCORE_FUNCTIONS:
            scores_df[score_func.__name__] = score_func(qualitative_matrix=self.qualitative_matrix, grid_clusters=self.grid_clusters)

        for score_func in PER_REACTION_SCORE_FUNCTIONS:
            func_name = score_func.__name__
            scores = []
            for rid in self.qualitative_matrix.columns:
                fva_reactions = self.modelclass.analyze.fva_reactions
                fva_results = self.modelclass.analyze.fva_results
                reaction_index = fva_reactions.index(rid)
                val = score_func(
                    reaction_states=self.qualitative_matrix[rid].values,
                    clusters=self.grid_clusters,
                    fva_result = (fva_results[:, reaction_index, :])
                )
                scores.append(val)
            scores_df[func_name] = scores

        col_to_sort = scores_df.columns[sort_index] if sort_index < len(scores_df.columns) else scores_df.columns[0]
        if sort_score:
            scores_df = scores_df.sort_values(by=col_to_sort, ascending=False)

        metric_names = [f.__name__ for f in GLOBAL_SCORE_FUNCTIONS] + [f.__name__ for f in PER_REACTION_SCORE_FUNCTIONS]
        rank_df, feature_selection = self._consensus_feature_selection(scores_df, metric_names, **kwargs)
        logging.info(f"Number of reactions selected: {len(feature_selection)}\n")

        return scores_df, rank_df, feature_selection
    

    @staticmethod
    def _consensus_feature_selection(
        score_df: pd.DataFrame, score_cols: list[str], top_T: int = 40, min_votes: int = 2, **kwargs,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Consensus feature selection via top-T voting across multiple scores.

        Parameters
        ----------
        score_df : pd.DataFrame
            Index = reaction_id, columns = scores
        score_cols : list of str
            Columns to use as scores
        top_T : int
            Top-T reactions per score
        min_votes : int or None
            Minimum number of appearances required. 

        Returns
        -------
        rank_df : DataFrame
            Per-metric rank positions (1 = best).
        selected_reactions : list[str]
            Reactions selected by consensus voting.
        """
        rank_df = pd.DataFrame(
            {col: score_df[col].rank(ascending=False, method="average") for col in score_cols},
            index=score_df.index,
        )

        topT_mask = rank_df <= top_T
        votes = topT_mask.sum(axis=1)

        rank_df["votes"] = votes
        rank_df = rank_df[votes >= min_votes]
        rank_df = rank_df.sort_values(["votes"], ascending=[False]) # type: ignore

        reaction_selection = list(rank_df.index)
        return rank_df, reaction_selection
    

    @staticmethod
    def compare_clusters(
        clusters_df: pd.DataFrame, cluster_id1: str | int, cluster_id2: str | int
    ) -> pd.DataFrame:
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

