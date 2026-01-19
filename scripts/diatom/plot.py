from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


if TYPE_CHECKING:
    from .diatom import Diatom


class DiatomPlot():
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom


    def plot_sampled_polytope(self, show_boundary: bool = True, show_points: bool = True,
        s: float = 10.0, alpha: float = 0.6) -> None:
        grid = self.diatom.grid
        analyze = self.diatom.analyze

        poly = analyze.polytope
        points = grid.points
        feasible = grid.feasible_points
        
        points = points[feasible]

        fig, ax = plt.subplots(figsize=(7, 7))

        # --- Polígono ---
        if show_boundary:
            assert isinstance(poly, Polygon)
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=1, linestyle="dashed")
  
        # --- Puntos sampleados ---
        if show_points and points.size > 0:

            clusters = self.diatom.clustering.grid_clusters
            if clusters is None:
                raise RuntimeError("Grid clusters not computed.")

            sc = ax.scatter(points[:, 0], points[:, 1], c=clusters, s=s, alpha=alpha, cmap = "tab20")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("cluster id")

        reactions = analyze.analyzed_reactions
        ax.set_xlabel(reactions[0])
        ax.set_ylabel(reactions[1])
        ax.set_title("Projected feasible polytope")
        ax.grid(True)

        plt.tight_layout()
        plt.show()


