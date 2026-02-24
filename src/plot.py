import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from .metabolic_experiment import MetabolicExperiment


class Plot():
    """Utility class for visualizing geometric, sampling, clustering and qualitative
    analysis results associated with the metabolic experiment.

    Parameters
    ----------
    parent_class : MetabolicExperiment
        Parent class providing access to the metabolic model, grid sampler, and I/O utilities.
    """
    def __init__(self, parent_class: "MetabolicExperiment"):
        self.parent_class = parent_class


    def polytope_shape(self, **kwargs):
        """Plot the projected feasible polytope together with the sampling grid.

        Note
        ----
        Intended mainly for debugging the geometry of the projection.
        """
        self.parent_class._require(set_instance=True, polytope=True)
        poly = self.parent_class.projection.polytope
        grid = self.parent_class.grid

        _, lines = grid._build_grid()

        fig, ax = plt.subplots(figsize=(5, 5))

        # grid lines 
        self._plot_geometry(ax, lines, color="red", alpha=0.3, linewidth=1)
        # polygon
        self._plot_geometry(ax, poly, color="black", linewidth=2)
        # boundary 
        self._plot_geometry(ax, poly.boundary, color="blue", linewidth=2)

        ax.legend()
        reaction1, reaction2 = self.parent_class.analyze.analyzed_reactions
        ax.set_xlabel(reaction1)
        ax.set_ylabel(reaction2)
        title = f"Projected Feasible Polytope Shape ({reaction1} - {reaction2})"
        for key, value in kwargs.items():
            title += f", {key}: {value}"
        ax.set_title(title)
        ax.grid(True)

        plt.tight_layout()
        plt.show()


    def _plot_geometry(self, ax, geom, **kwargs):
        """Helper method to plot Shapely geometries on a Matplotlib axis."""
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.plot(x, y, **kwargs)

        elif geom.geom_type == "LineString":
            x, y = geom.xy
            ax.plot(x, y, **kwargs)

        elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
            for g in geom.geoms:
                self._plot_geometry(ax, g, **kwargs)


    def sampled_polytope(
        self, 
        show_boundary: bool = False, 
        show_points: bool = True,
        s: float = 15.0, 
        **kwargs,
    ) -> None:
        """Plot sampled grid points inside the feasible polytope, colored by cluster.

        Parameters
        ----------
        show_boundary : bool, default=False
            Whether to draw the polytope boundary.
        show_points : bool, default=True
            Whether to draw sampled grid points.
        s : float, default=15.0
            Marker size for sampled points.
        kwargs :
            Extra parameters to be used in the plot title.
        """
        self.parent_class._require(polytope=True, grid_points=True, clusters=True)
      
        poly = self.parent_class.projection.polytope
        
        grid = self.parent_class.grid
        points = grid.points
        analyzed = grid.analyzed_points

        clusters = self.parent_class.clustering.clusters

        points = points[analyzed]

        fig, ax = plt.subplots(figsize=(7, 6))

        if show_boundary:
            assert isinstance(poly, Polygon)
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=1, linestyle="dashed", alpha=0.5, color="gray")
  
        if show_points and points.size > 0:
            k = int(np.max(clusters))
            cmap = plt.get_cmap("tab20", k)
            vmin = 0.5
            vmax = k + 0.5

            sc = ax.scatter(points[:, 0], points[:, 1], c=clusters, s=s, alpha=1.0, 
                            cmap = cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(1, k+1))
            cbar.ax.set_yticklabels([f"C{i}" for i in range(1, k+1)])
            cbar.set_label("Cluster ID")

        reaction1, reaction2 = self.parent_class.analyze.analyzed_reactions
        ax.set_xlabel(reaction1)
        ax.set_ylabel(reaction2)
        title = f"Feasible Polytope Sampling ({reaction1} - {reaction2})"
        for key, value in kwargs.items():
            title += f", {key}: {value}"
        ax.set_title(title)
        ax.grid(True)

        plt.tight_layout()
        
        path = self.parent_class.io.save_plot_path()
        if path is not None:
            plt.savefig(path)
            
        plt.show()


    def cluster_distribution(self, clusters_df: pd.DataFrame, cmap: str = 'Accent', figsize: tuple[int, int] = (10,5)) -> pd.DataFrame:
        """Plot the distribution of qualitative reaction categories per cluster.

        Each column in `clusters_df` is interpreted as a cluster and each row as a reaction category assignment. 
        NaN values are treated as an additional 'variable' category. 

        Parameters
        ----------
        clusters_df : pd.DataFrame
            DataFrame where each column corresponds to a cluster and each row
            to a reaction. Values are qualitative reaction categories.
        cmap : str, default='Accent'
            Matplotlib colormap used for the stacked bar plot.
        figsize : tuple[int, int], default=(10, 5)
            Figure size.

        Returns
        -------
        pd.DataFrame
            DataFrame containing category percentages per cluster.
        """
        cluster_columns = list(clusters_df)
        n_reactions = clusters_df.shape[0]

        nan_rep = 100.0 # Change nan to additional category 'variable'
        filled_clusters_df = clusters_df.fillna(nan_rep)

        category_percents_dict = dict()

        for category in cluster_columns:
            vc = filled_clusters_df[category].value_counts()
            vc = 100 * vc/n_reactions # to percentages
            category_percents_dict[category] = vc.to_dict()   
            
        category_percents = pd.DataFrame.from_dict(category_percents_dict, orient='index')
        category_percents.fillna(0, inplace=True)
        category_percents.rename(columns = self.parent_class.analyze.category_dict, inplace=True)

        # plot
        ax = category_percents.plot.barh(stacked=True, cmap=cmap, figsize=figsize)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='reaction category')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel('clusters')
        ax.set_xlabel('reactions')

        return category_percents 


    def dendrogram(self):
        """Plots the dendrogram produced by hierarchical clustering."""
        from scipy.cluster.hierarchy import dendrogram as dendro
        self.parent_class._require(clusters=True)

        plt.figure(figsize=(10, 5))
        plt.title('Hierarchical Clustering Dendrogram')
        dendro(
            self.parent_class.clustering.linkage_matrix,
            #truncate_mode='lastp',
            #p=8,
            show_leaf_counts=True,
            leaf_rotation=90,
            leaf_font_size=9,
        )
        plt.show() 


    def qFCA_results(self, col_wrap: int = 4) -> None:
        """Plot quantitative FCA (qFCA) results. For each sampled point, the region 
        between minimum and maximum FVA curves is visualized.

        Parameters
        ----------
        col_wrap : int, default=4
            Number of subplot columns before wrapping.
        """
        self.parent_class._require(qfca=True)
        maxmin_df = self.parent_class.analyze.qFCA

        sns.set(font_scale = 2)
        rxns_analysis = maxmin_df.columns[0:2]
        sns.set_style("whitegrid")

        g=sns.relplot(data = maxmin_df, x=rxns_analysis[0], y=rxns_analysis[1], col = 'point', hue='FVA', kind='line', col_wrap=col_wrap, lw=0)
        points = maxmin_df.point.unique()
        for i,ax in enumerate(g.axes):
            p = points[i]

            p_df = maxmin_df.loc[maxmin_df['point']==p]
            x = p_df.loc[p_df['FVA']=='maximum'][rxns_analysis[0]].to_numpy()

            y1 = p_df.loc[p_df['FVA']=='maximum']
            y1 = y1[rxns_analysis[1]].to_numpy()

            y2 = p_df.loc[p_df['FVA']=='minimum']
            y2 = y2[rxns_analysis[1]].to_numpy()

            ax.fill_between(x, y1,y2, color='none',hatch='//', edgecolor="k", linewidth=0.001)


    def reaction_scores(self, df: pd.DataFrame, column: str, top_n: int = 25):
        """Plot the top-N reactions ranked by a given score.

        Assumes `df` is already sorted in descending order by `column`.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing reaction scores.
        column : str
            Column name used for ranking.
        top_n : int, default=25
            Number of top reactions to display.
        """
        plt.figure(figsize=(10, 6))
        # Tomamos el top N
        data_to_plot = df[column].head(top_n)
        
        sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette="viridis")
        
        plt.title(f"Top {top_n} reactions based on {column}")
        plt.xlabel(f"Score ({column})")
        plt.ylabel("Reaction ID")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

