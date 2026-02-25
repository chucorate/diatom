import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from cobra import Reaction

from src.metrics import REACTION_METRIC_LIST, GLOBAL_METRIC_LIST, Floating, ratio_metric
from src.feature_selection import PER_REACTION_SCORE_FUNCTIONS, GLOBAL_SCORE_FUNCTIONS

if TYPE_CHECKING:
    from .metabolic_experiment import MetabolicExperiment


class Clustering():
    """Class for managing clustering grid points and reactions based on qualitative FVA profiles,
    and for computing cluster-level summaries, metrics, and reaction scores.
    
    Parameters
    ----------
    parent_class : MetabolicExperiment
        Parent class providing access to the metabolic model, grid sampler, and I/O utilities.

    Attributes
    ----------
    n_clusters : int
        Number of clusters produced.

    clusters : np.ndarray, shape (n_points, )
        Array containing the cluster labels of all grid points. 

    linkage_matrix : np.ndarray, shape (n_points-1, 4)
        Linkage matrix encoding the dendrogram produced via hierarchical clustering.

    representatives : pd.DataFrame
        Rows correspond to reactions and columns to clusters (``c1, c2, ...``).
        Entries are representative qualitative values or NaN, obtained by clustering.

    reaction_metrics : list[Callable]
        List of metrics that get applied to individual reactions.

    global_metrics: list[Callable]
        List of metrics that get applied to the whole set of reactions analyzed.

    reaction_score_metrics: list[Callable]
        List of score metrics used for feature selection. They depend on reactions
        at the individual level.

    global_score_metrics: list[Callable]
        List of score metrics used for feature selection. They evaluate the whole
        qualitative matrix.
    """
    def __init__(self, parent_class: "MetabolicExperiment"):
        self.parent_class = parent_class

        self.n_clusters: int 
        self.clusters: np.ndarray 
        self.linkage_matrix: np.ndarray 
        self.representatives: pd.DataFrame

        self.reaction_metrics: list[Callable] = REACTION_METRIC_LIST
        self.global_metrics: list[Callable] = GLOBAL_METRIC_LIST
        self.reaction_score_metrics: list[Callable] = PER_REACTION_SCORE_FUNCTIONS
        self.global_score_metrics: list[Callable] = GLOBAL_SCORE_FUNCTIONS
        

    @property
    def qualitative_matrix(self) -> pd.DataFrame:
        return self.parent_class.analyze.qualitative_matrix


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
        

    def set_clusters(self, n_clusters: int, linkage_method: str, changing: bool = True, **kwargs) -> None:
        """Cluster grid points based on qualitative flux vectors.

        Uses pairwise Jaccard distances between grid points and stores the resulting
        cluster labels and number of clusters as attributes.

        Parameters
        ----------
        n_clusters : int
            Target number of clusters.
        linkage_method : str
            Linkage method to be used by hierarchical clustering.
        changing : bool
            If True, restrict to reactions that change across the grid.
        
        Attributes Set
        --------------
        n_clusters : int
            Number of clusters produced.
        clusters : np.ndarray, shape (n_points, )
            Array containing the cluster labels of all grid points. 
        linkage_matrix : np.ndarray, shape (n_points-1, 4)
            Linkage matrix encoding the dendrogram produced via hierarchical clustering.
        """
        self.parent_class._require(qualitative_matrix=True)

        qualitative_vector = self.one_hot_encode_reactions(changing)
        
        logging.info("Clustering grid points ...") 
        self.n_clusters, self.clusters, self.linkage_matrix = (
            self._map_clusters(qualitative_vector, n_clusters, linkage_method, **kwargs)
        )
        self.parent_class.io.save_clusters(self.n_clusters, self.clusters)


    @staticmethod
    def _map_clusters(
        qualitative_vector: np.ndarray, 
        n_clusters: int, 
        linkage_method: str, 
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

        linkage_matrix = hierarchy.linkage(dvector, method=linkage_method)
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion=criterion, **kwargs) # clusters are indexed from 1

        n_clusters = len(np.unique(clusters))
        
        logging.info(f"Done! Obtained {n_clusters} from hierarchical clustering.")
        logging.debug(f"Linkage method: {linkage_method}, criterion: {criterion}, metric: {metric}.")    
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


    def get_cluster_qualitative_profiles(
        self, 
        threshold: float = 0.80,
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
        threshold : float, default=0.80
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
        self.parent_class._require(qualitative_matrix=True, clusters=True)
        
        vector_df = self.qualitative_matrix.astype('int32')

        cluster_ids = np.arange(1, self.n_clusters + 1)
        cluster_dfs = [vector_df[self.clusters == cluster_id] for cluster_id in cluster_ids]
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

        analyze = self.parent_class.analyze

        if changing:
            changing_filter = representatives.apply(lambda x: x.unique().size > 1, axis = 1)    
            representatives = representatives[changing_filter]
        
        if selected_reactions:
            representatives = representatives.loc[selected_reactions] 

        representatives = representatives.replace(analyze.category_dict)
       
        self.parent_class.io.save_cluster_df(
            representatives, 
            "Qualitative_profiles", 
            reaction_len=len(selected_reactions) if selected_reactions is not None else -1, 
            index=True, 
            overwrite=overwrite,
        )
        self.representatives = representatives
        return representatives.reset_index(names="reaction id")
    

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
        self.parent_class._require(clusters=True, qualitative_matrix=True)

        reaction_list = self.representatives.index.tolist()

        n_points = self.qualitative_matrix.shape[0]
        qual_states = self.qualitative_matrix[reaction_list].replace(self.parent_class.analyze.category_dict)
        
        n_clusters = self.n_clusters
        cluster_masks = {cluster_id: self.clusters == cluster_id for cluster_id in range(1, n_clusters+1)}
    
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
        self.parent_class._require(clusters=True)
        
        clusters = self.clusters
        fva_reactions = self.parent_class.analyze.fva_reactions
        fva_results = self.parent_class.analyze.fva_results

        metric_names = [metric.__name__ for metric in self.global_metrics]

        rows: list[dict[str, Any]] = []
        for cluster_index in range(1, self.n_clusters+1):
            metric_results = [
                metric(fva_reactions, fva_results, clusters, cluster_index) 
                for metric in self.global_metrics
            ]

            for metric_name, metric_value in zip(metric_names, metric_results):
                rows.append({"cluster": cluster_index, "metric": metric_name, "value": metric_value})

        df = pd.DataFrame(rows)
        self.parent_class.io.save_cluster_df(
            df, 
            "Global_metrics", 
            reaction_len=len(reaction_list), 
            metric_list=self.global_metrics, 
            overwrite=overwrite,
        )
        return df
    

    def get_cluster_metrics_per_reaction(
        self, reaction_list: list[str], overwrite: bool = False,
    ) -> pd.DataFrame:
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
        self.parent_class._require(clusters=True)

        clusters = self.clusters
        fva_reactions = self.parent_class.analyze.fva_reactions
        fva_results = self.parent_class.analyze.fva_results

        metric_names = [metric.__name__ for metric in self.reaction_metrics]

        rows: list[dict[str, Any]] = []

        for reaction_id in reaction_list:
            reaction_index = fva_reactions.index(reaction_id)
            reaction_fva_results = (fva_results[:, reaction_index, :])

            for cluster_index in range(1, self.n_clusters+1):
                filtered_results = reaction_fva_results[clusters == cluster_index]
                metric_results = [metric(filtered_results) for metric in self.reaction_metrics]

                for metric_name, metric_value in zip(metric_names, metric_results):
                    rows.append({
                        "reaction_id": reaction_id,
                        "cluster": cluster_index,
                        "metric": metric_name,
                        "value": metric_value
                    })

        df = pd.DataFrame(rows)
        self.parent_class.io.save_cluster_df(
            df, 
            "Metrics_per_reaction", 
            reaction_len=len(reaction_list), 
            metric_list=self.reaction_metrics, 
            overwrite=overwrite,
        )
        return df


    def get_reactions_dataframe(self, reaction_list: list[str]) -> pd.DataFrame:
        "Creates a dataframe with relevant information about every reaction given."
        rows = []
        for reaction_id in reaction_list:
            reaction = cast(Reaction, self.parent_class.model.reactions.get_by_id(reaction_id))
            rows.append({
                "Reaction ID": reaction_id,
                "Full Name": reaction.name,
                "Reaction": reaction.reaction,
                "Metabolite List": set([metabolite.name for metabolite in reaction.metabolites.keys()])
            })

        reactions_df = pd.DataFrame(rows)
        return reactions_df

    
    def reaction_scores(
        self, sort_score: bool = True, sort_index: int = 0, top_T: int = 20, **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Compute reaction-level scores and perform consensus feature selection.

        This method evaluates multiple scoring functions at the reaction level,
        combining global (cluster-wise) and per-reaction criteria. The resulting
        score matrix is then used to select a subset of reactions via consensus
        voting.

        Parameters
        ----------
        sort_score : bool, default=True
            If True, the score DataFrame is sorted in descending order according
            to the metric specified by `sort_index`.

        sort_index : int, default=0
            Index of the score metric (column) used to sort the score DataFrame.
            If the index is out of range, the first metric is used.

        top_T : int, default=20
            Number of top-ranked reactions considered by each metric during
            consensus feature selection.

        **kwargs
            Additional keyword arguments passed to `_consensus_feature_selection`.

        Returns
        -------
        score_df : pd.DataFrame
            DataFrame indexed by reaction ID. Columns correspond to score functions from 
            `global_score_metrics` and `reaction_score_metrics`.
        rank_df : pd.DataFrame
            DataFrame indexed by reaction ID, containing how well each reaction performed
            against all considered metrics (1 = best).
        selected_reactions : list[str]
            Reactions selected via consensus voting.

        Notes
        -----
        - Global score functions operate on the full qualitative vector.
        - Per-reaction score functions additionally depend on reaction-specific FVA results.
        """
        self.parent_class._require(clusters=True, qualitative_matrix=True)

        fva_reactions = self.parent_class.analyze.fva_reactions
        fva_results = self.parent_class.analyze.fva_results

        scores_df = pd.DataFrame(index=self.qualitative_matrix.columns)

        for score_func in self.global_score_metrics:
            scores_df[score_func.__name__] = score_func(
                qualitative_matrix=self.qualitative_matrix, 
                clusters=self.clusters,
            )

        for score_func in self.reaction_score_metrics:
            func_name = score_func.__name__
            scores = []
            for rid in self.qualitative_matrix.columns:
                reaction_index = fva_reactions.index(rid)
                val = score_func(
                    reaction_states=self.qualitative_matrix[rid].values,
                    clusters=self.clusters,
                    fva_result = (fva_results[:, reaction_index, :])
                )
                scores.append(val)
            scores_df[func_name] = scores

        col_to_sort = scores_df.columns[sort_index] if sort_index < len(scores_df.columns) else scores_df.columns[0]
        if sort_score:
            scores_df = scores_df.sort_values(by=col_to_sort, ascending=False)

        metric_names = (
            [f.__name__ for f in self.global_score_metrics] + [f.__name__ for f in self.reaction_score_metrics]
        )
        rank_df, feature_selection = self._consensus_feature_selection(
            scores_df, metric_names, top_T=top_T, **kwargs,
        )
        logging.info(f"Number of reactions selected: {len(feature_selection)}\n")

        return scores_df, rank_df, feature_selection
    

    @staticmethod
    def _consensus_feature_selection(
        score_df: pd.DataFrame, score_cols: list[str], top_T: int, min_votes: int = 2, **kwargs,
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
        rank_df : pd.DataFrame
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


    def set_ratio_metric(self,
        metric_name: str,
        numerator: str | list[str],
        denominator: str | list[str],
        num_func: Callable[[Floating, Floating], Floating] | None = None,
        den_func: Callable[[Floating, Floating], Floating] | None = None,
        add_to_metrics: bool = True,
    ) -> None:
        """
        Defines and registers a custom ratio-based global metric.

        For each reaction in the numerator and denominator sets, this metric:
        1. Extracts the FVA min/max values restricted to the specified cluster.
        2. Computes the midpoint of the feasible interval at each grid point.
        3. Takes the median midpoint across points.
        4. Computes the absolute value of that median.
        5. Sums contributions across all reactions in the set.

        The final metric is computed as a ratio between the aggregated numerator
        and denominator values. Optional transformation functions can be applied
        to the numerator and/or denominator before division.

        Parameters
        ----------
        metric_name : str
            Name assigned to the generated metric function. This name is used
            for identification and reporting.
        numerator : str or list[str]
            Reaction ID or list of reaction IDs defining the numerator term.
        denominator : str or list[str]
            Reaction ID or list of reaction IDs defining the denominator term.
        num_func : callable, optional
            Transformation applied to the aggregated numerator value
            (receives numerator and denominator aggregates as input).
        den_func : callable, optional
            Transformation applied to the aggregated denominator value
            (receives numerator and denominator aggregates as input).
        add_to_metrics : bool, default=True
            If True, the generated metric is appended to `self.global_metrics`.

        Attributes Set
        -------
        The method registers the metric in `self.global_metrics` if configured to do so.
        """
        def metric(
            fva_reactions: list[str], fva_results: np.ndarray, clusters: np.ndarray, cluster_index: int
        ) -> float:
            ratio = ratio_metric(
                fva_reactions, 
                fva_results, 
                clusters, 
                cluster_index, 
                numerator,
                denominator,
                num_func=num_func,
                den_func=den_func,
            )
            return ratio

        metric.__name__ = metric_name

        if add_to_metrics:
            self.global_metrics.append(metric)

    
    def show_all_metrics(self) -> None:
        all_metrics_dict = {
            "Reaction Metrics": self.reaction_metrics, 
            "Global Metrics": self.global_metrics, 
            "Feature Selection Metrics": self.reaction_score_metrics + self.global_score_metrics, 
        }

        for list_name, metric_list in all_metrics_dict.items():
            print(f"{list_name}:\n{[metric.__name__ for metric in metric_list]}\n")

