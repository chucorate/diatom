from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from .diatom import Diatom


class DiatomPlot():
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom


    def sampled_polytope(self, show_boundary: bool = True, show_points: bool = True,
        s: float = 12.0, alpha: float = 1.0) -> None:
        clusters = self.diatom.clustering.grid_clusters
        if clusters is None:
            raise RuntimeError("Grid clusters not computed.")
        
        grid = self.diatom.grid
        analyze = self.diatom.analyze

        poly = analyze.polytope
        points = grid.points
        analyzed = grid.analyzed_points
        
        points = points[analyzed]

        fig, ax = plt.subplots(figsize=(7, 7))

        # --- Polígono ---
        if show_boundary:
            assert isinstance(poly, Polygon)
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=1, linestyle="dashed", alpha=0.5, color="gray")
  
        # --- Puntos sampleados ---
        if show_points and points.size > 0:
            k = int(np.max(clusters))
            cmap = plt.get_cmap("tab20", k)
            vmin = 0.5
            vmax = k + 0.5

            sc = ax.scatter(points[:, 0], points[:, 1], c=clusters, s=s, alpha=alpha, 
                            cmap = cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(1, k+1))
            cbar.ax.set_yticklabels([f"C{i}" for i in range(1, k+1)])
            cbar.set_label("Cluster ID")

        reaction1, reaction2 = analyze.analyzed_reactions

        ax.set_xlabel(reaction1)
        ax.set_ylabel(reaction2)
        ax.set_title(f"Projected Feasible Polytope ({reaction1} - {reaction2})")
        ax.grid(True)

        plt.tight_layout()
        #ax.set_aspect("auto")
        #ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        #ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        #fig.subplots_adjust(right=0.88)

        plt.savefig(self.diatom.io.save_plot_path())
        plt.show()


    def plot_cluster_distribution(self, clusters_df: pd.DataFrame, cmap: str = 'Accent', figsize: tuple[int, int] = (10,5)) -> pd.DataFrame:
        """Plot the distribution of qualitative reaction categories per cluster.

        Each column in `clusters_df` is interpreted as a cluster and each row as a
        reaction category assignment. NaN values are treated as an additional
        'variable' category. Outputs a dataframe contaning the percentage of reactions per category for each cluster.
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
        category_percents.rename(columns = self.diatom.analyze.category_dict, inplace=True)

        # plot
        ax = category_percents.plot.barh(stacked=True, cmap=cmap, figsize=figsize)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='reaction category')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel('clusters')
        ax.set_xlabel('reactions')

        return category_percents 


    def plot_qFCA(self, col_wrap = 4):
        #Plots results computed by quan_FCA
        #input: maxmin_df (output of quan_FCA)
        #output: plot
        maxmin_df = self.diatom.analyze.qFCA

        sns.set(font_scale = 2)
        rxns_analysis = maxmin_df.columns[0:2]
        sns.set_style("whitegrid")

        g=sns.relplot(data = maxmin_df, x=rxns_analysis[0], y=rxns_analysis[1], col = 'point', hue='FVA', kind='line', col_wrap=4, lw=0)
        points = maxmin_df.point.unique()
        for i,ax in enumerate(g._axes):
            p = points[i]

            p_df = maxmin_df.loc[maxmin_df['point']==p]
            x = p_df.loc[p_df['FVA']=='maximum'][rxns_analysis[0]].to_numpy()

            y1 = p_df.loc[p_df['FVA']=='maximum']
            y1 = y1[rxns_analysis[1]].to_numpy()

            y2 = p_df.loc[p_df['FVA']=='minimum']
            y2 = y2[rxns_analysis[1]].to_numpy()

            ax.fill_between(x, y1,y2, color='none',hatch='//', edgecolor="k", linewidth=0.001)