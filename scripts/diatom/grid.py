from typing import TYPE_CHECKING, cast
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from shapely import intersection_all
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep
from shapely.ops import unary_union

if TYPE_CHECKING:
    from .diatom import Diatom


class DiatomGrid():
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom
        self.points: NDArray[np.floating] = np.array([]) # shape: (numPoints**2, 2)
        self.feasible_points: NDArray[np.bool] = np.array([])
        self.grid_dimensions: NDArray[np.floating] = np.array([0,0])
        self.points_per_axis: tuple[int, int] = (0,0)


    def _build_grid(self, delta: float = 0.1) -> tuple[NDArray[np.floating], BaseGeometry]:
        poly = self.diatom.analyze.polytope

        minx, miny, maxx, maxy = poly.bounds
        interval_x = maxx - minx
        interval_y = maxy - miny
        self.grid_dimensions = np.array([interval_x, interval_y])

        dx, dy = delta * interval_x, delta * interval_y

        xs = np.arange(minx, maxx + dx, dx)
        ys = np.arange(miny, maxy + dy, dy)

        self.points_per_axis = (len(xs), len(ys))

        lines = []
        for x in xs:
            lines.append(LineString([(x, miny), (x, maxy)]))
        for y in ys:
            lines.append(LineString([(minx, y), (maxx, y)]))

        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        return grid_points, unary_union(lines)


    @staticmethod
    def _iter_geoms(g: BaseGeometry):
        if isinstance(g, GeometryCollection):
            yield from g.geoms
        elif isinstance(g, MultiPoint):
            yield from g.geoms
        else:
            yield g


    def _intersection_points(self, lines: BaseGeometry) -> NDArray[np.floating]:
        poly = self.diatom.analyze.polytope
        inter = intersection_all([lines, poly.boundary])

        points = [(g.x, g.y) for g in self._iter_geoms(inter) if isinstance(g, Point)]

        return np.asarray(points, dtype=float)


    def sample_polytope(self, delta: float = 0.1, eps = 1e-9) -> None:
        poly = self.diatom.analyze.polytope
        prepared_poly = prep(poly.buffer(eps))

        covers_xy = getattr(prepared_poly, "covers_xy", None)

        grid_points, lines = self._build_grid(delta=delta)
        intersect_points = self._intersection_points(lines)
        poly_vertices = np.asarray(poly.exterior.coords, dtype=float)[:-1] # type: ignore

        points = np.vstack([grid_points, intersect_points, poly_vertices])
        points = np.unique(points, axis=0)

        self.points = points

        if covers_xy is not None:
            feasible = np.fromiter(
                (covers_xy(x, y) for x, y in points),
                dtype=bool,
                count=len(points)
            )
        else:
            feasible = np.fromiter(
                (prepared_poly.covers(Point(x, y)) for x, y in points),
                dtype=bool,
                count=len(points)
                )

        self.feasible_points = feasible

    
    

    def debug_plot(self, delta: float = 0.1):
        poly = self.diatom.analyze.polytope

        grid_points, lines = self._build_grid(delta)
        inter_points = self._intersection_points(lines)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Grid lines (rojo claro)
        plot_geometry(ax, lines, color="red", alpha=0.3, linewidth=1)

        # Polígono
        plot_geometry(ax, poly, color="black", linewidth=2)

        # Borde (azul)
        plot_geometry(ax, poly.boundary, color="blue", linewidth=2)

        # Intersecciones
        if len(inter_points) > 0:
            ax.scatter(
                inter_points[:, 0],
                inter_points[:, 1],
                color="yellow",
                edgecolor="black",
                s=60,
                zorder=10,
                label="Intersections"
            )

        ax.legend()
        plt.show()


def plot_geometry(ax, geom, **kwargs):
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, **kwargs)

    elif geom.geom_type == "LineString":
        x, y = geom.xy
        ax.plot(x, y, **kwargs)

    elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            plot_geometry(ax, g, **kwargs)