from typing import TYPE_CHECKING, Callable, Any

import numpy as np
import pandas as pd

from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy

from scripts.metrics import REACTION_METRIC_LIST, GLOBAL_METRIC_LIST

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem
    from diatom.diatom import Diatom


class ModelClustering():
    def __init__(self, modelclass: "BaseEcosystem | Diatom"):
        self.modelclass = modelclass
        self.initial_n_clusters: int = 0
        self.grid_n_clusters: int = 0
        self.reaction_n_clusters: int = 0
        self.grid_clusters, self.reaction_clusters = None, None
        self.bin_vector_df = None


    @property
    def qual_vector_df(self) -> pd.DataFrame:
        return self.modelclass.analyze.qual_vector_df


    def _filter_changing_reactions(self, changing: bool = True) -> np.ndarray:
        z = self.qual_vector_df.copy()

        if not changing:
            return z.values
        
        #self.changed_rxns = None
        changed_rxns = self.qual_vector_df.max(axis=0) != self.qual_vector_df.min(axis=0)
        changed_rxns_ids = z.columns[changed_rxns]
        z = z[changed_rxns_ids]
        #self.changed_rxns = changed_rxns_ids
        #print(f"changing reaction: {self.qual_vector_df.values.shape} -> {changed_rxns.sum()}")

        z_one_hot = pd.get_dummies(z.astype(str))
        print(f"base: {z.shape} -> one-hot: {z_one_hot.shape}")
        return z_one_hot.values
        

    # NO SE USA EN NINGUN MODULO
    def set_reaction_clusters(self, method, changing: bool = True, **kwargs) -> None:
        """Cluster reactions based on their qualitative FVA profiles across grid points.

        Optionally restricts clustering to reactions whose qualitative state changes
        across the grid. Stores cluster labels and number of clusters as attributes.
        """
        if self.qual_vector_df is None:
            print("No qualitative FVA values stored. Run qual_fva analysis first!")
            return
        
        z = self._filter_changing_reactions(changing)
        z = z.T

        print(f"Clustering {z.shape[0]} reactions ...") 
        self.reaction_n_clusters, self.reaction_clusters = self._map_clusters(method, z, **kwargs)

 
    def set_grid_clusters(
            self, 
            method: str, 
            changing: bool = True, 
            vector: str = 'qual_vector', 
            initial_n_clusters: int = 20, 
            **kwargs
        ) -> None:
        """Cluster grid points based on qualitative or binary flux vectors.

        Uses pairwise Jaccard distances between grid points and stores the resulting
        cluster labels and number of clusters as attributes.
        """
        self.initial_n_clusters = initial_n_clusters
        
        loaded_clusters = self.modelclass.io.load_clusters()
        if isinstance(loaded_clusters, tuple):
            self.grid_n_clusters, self.grid_clusters = loaded_clusters
            return 

        if self.qual_vector_df is None:
            print("No qualitative FVA values stored. Run qual_fva analysis first!")
            return

        #NJT to use qual_vector as well as bin_vector if required
        if vector == 'qual_vector':
            qualitative_vector = self._filter_changing_reactions(changing)
        elif vector == 'bin_vector' and self.bin_vector_df is not None:
            qualitative_vector = self.bin_vector_df.values
        else:
            raise ValueError(f"Unknown vector: {vector}")

        print("Clustering grid points ...") 
        self.grid_n_clusters, self.grid_clusters = self._map_clusters(method, qualitative_vector, **kwargs)
        self.modelclass.io.save_clusters(self.grid_n_clusters, self.grid_clusters)


    def _map_clusters(self, method: str, qualitative_vector: np.ndarray, **kwargs) -> tuple[int, np.ndarray]:
        """Compute pairwise Jaccard distances and apply a clustering method.

        Returns the number of clusters and a vector of cluster labels.
        """
        distance_metric = 'jaccard'
        dvector = distance.pdist(qualitative_vector, distance_metric) 
        dmatrix = distance.squareform(dvector)     

        if method == 'hierarchical':
            n_clusters, clusters = get_hierarchical_clusters(dvector,**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Done! n_clusters: {n_clusters}")    
        return n_clusters, clusters


    #function to get representative qualitative values of a reaction in a cluster
    @staticmethod
    def _get_representative_qualitative_values(cluster_column: pd.Series, threshold: float) -> int | None:
        """Return the dominant qualitative value in a cluster if it exceeds a frequency threshold.
        If the threshold is not met, returns None."""
        total = len(cluster_column)

        qualitative_values, counts = np.unique(cluster_column, return_counts=True)
        #print(f'thresholds: {counts/total}, qualitative_values: {qualitative_values, counts, total}')
        representative = qualitative_values[counts/total >= threshold]
         
        if representative.size > 0:
            return representative[0] # qualitative value present if at least threshold of reactions in cluster  
        
        return None    


    def get_grid_cluster_qual_profiles(
            self, 
            vector: str = 'qual_vector', 
            threshold: float = 0.75,
            changing: bool = True, 
            convert: bool = True,
            selected_reactions: list[str] | None = None,
        ) -> pd.DataFrame:
        """Compute representative qualitative reaction profiles for each grid cluster.

        For each grid cluster, assigns a qualitative value to each reaction if it
        appears in at least a given fraction of grid points. Optionally filters
        reactions that change between clusters and converts qualitative codes.
        """
        if self.grid_clusters is None:
            raise RuntimeError("Missing clustering/qualitative FVA results!")
        
        #NJT to use qual_vector as well as bin_vector if required
        if vector =='qual_vector'and self.qual_vector_df is not None:
            vector_df = self.qual_vector_df
        elif vector == 'bin_vector' and self.bin_vector_df is not None:
            vector_df = self.bin_vector_df
        else:
            raise ValueError(f"Unknown vector: {vector}")
        
        vector_df = vector_df.astype('int32')
        #vector_df.head(200)
        cluster_ids = np.arange(1, self.grid_n_clusters + 1)
        #print(f"cluster_ids: {cluster_ids}, grid_clusters: {self.grid_clusters}")
        cluster_dfs = [vector_df[self.grid_clusters == cluster_id] for cluster_id in cluster_ids]
        print(f"cluster_dfs len: {len(cluster_dfs)}")
        representatives_list = [cluster_df.apply(self._get_representative_qualitative_values, 
                                                 threshold=threshold) for cluster_df in cluster_dfs]
        
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
       
        self.modelclass.io.save_cluster_df(representatives, "Qualitative_profiles", reaction_len=reaction_len, index=True)
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
    

    def get_cluster_global_metrics(self, reaction_list: list[str]) -> pd.DataFrame:
        grid_clusters = self.grid_clusters
        assert grid_clusters is not None

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
        self.modelclass.io.save_cluster_df(df, "Global_metrics", reaction_len=len(reaction_list), metric_list=GLOBAL_METRIC_LIST)
        return df
    

    def get_cluster_metrics_per_reaction(self, reaction_list: list[str]) -> pd.DataFrame:
        grid_clusters = self.grid_clusters
        assert grid_clusters is not None

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
        self.modelclass.io.save_cluster_df(df, "Metrics_per_reaction", reaction_len=len(reaction_list), metric_list=REACTION_METRIC_LIST)
        return df


# ======================================================= CLUSTER FUNCTIONS =======================================================


def get_hierarchical_clusters(dvector: np.ndarray, k: int = 20, lmethod: str = 'complete', 
                            criterion: str= 'maxclust', **kwards) -> tuple[int, np.ndarray]:
    row_linkage = hierarchy.linkage(dvector, method=lmethod)
    clusters = fcluster(row_linkage, t=k, criterion=criterion)

    k = len(np.unique(clusters))

    return k, clusters # clusters are indexed from 1

