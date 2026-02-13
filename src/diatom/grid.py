from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from shapely import intersection_all
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep
from shapely.ops import unary_union

if TYPE_CHECKING:
    from .diatom import Diatom


class DiatomGrid():
    """Grid-based sampler and feasibility evaluator over an existing 2D polytope.

    This class manages the construction of a regular 2D grid over the
    parameter space defined by a polytope, augments it with boundary-aware
    sampling (line intersections and vertices), and evaluates point feasibility.

    It is used to discretize the feasible region of the diatom parameter space 
    for qualitative and quantitative analyses (e.g. FVA, clustering).

    Parameters
    ----------
    diatom : Diatom
        Parent diatom object providing access to the analyzed polytope
        geometry and analysis pipelines.

    Attributes
    ----------
    points : np.ndarray, shape (n_points, 2)
        All sampled points in the parameter space, including grid points,
        boundary intersection points, and polytope vertices.

    feasible_points : np.ndarray, shape (n_points,)
        Boolean mask indicating whether each point in `points` lies inside
        or on the boundary of the polytope.

    analyzed_points : np.ndarray, shape (n_points,)
        Boolean mask indicating which points have been selected 
        for analysis. Subset of `feasible_points`.

    grid_dimensions : np.ndarray, shape (2,)
        Width and height of the polytope bounding box.

    points_per_axis : tuple[int, int]
        Number of grid points generated along the x and y axes.

    delta : float
        Relative grid spacing used to generate the sampling grid, expressed
        as a fraction of the polytope bounding box size.

    Notes
    -----
    The grid is constructed in normalized polytope coordinates using a
    relative spacing (`delta`). Boundary accuracy is improved by explicitly
    sampling line–boundary intersections and exact polytope vertices.

    Feasibility checks rely on Shapely prepared geometries and use
    `covers_xy` when available for performance.
    """
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom
        self.points: NDArray[np.floating] = np.array([]) # shape: (numPoints**2, 2)
        self.feasible_points: NDArray[np.bool] = np.array([])
        self.analyzed_points: NDArray[np.bool] = np.array([])
        self.grid_dimensions: NDArray[np.floating] = np.array([0,0])
        self.points_per_axis: tuple[int, int] = (0,0)
        self.grid_delta: float = 0.0


    @staticmethod
    def _iter_geoms(g: BaseGeometry):
        """Iterator for point class handling."""
        if isinstance(g, GeometryCollection):
            yield from g.geoms
        elif isinstance(g, MultiPoint):
            yield from g.geoms
        else:
            yield g


    def _intersection_points(self, lines: BaseGeometry) -> NDArray[np.floating]:
        """Gets all the points that intersect the polytope and the grid sampling lines."""
        poly = self.diatom.projection.polytope
        inter = intersection_all([lines, poly.boundary])

        points = [(g.x, g.y) for g in self._iter_geoms(inter) if isinstance(g, Point)]

        return np.asarray(points, dtype=float)


    def sample_polytope(self, eps: float = 1e-8) -> None:
        """Sample candidate points inside and on the boundary of a 2D polytope.

        Constructs a set of points by combining:
        - A regular grid over the parameter space.
        - Intersection points between grid lines and the polytope boundary.
        - Exact vertices of the polytope.

        All points are deduplicated and then tested for feasibility, defined as
        being covered by the polytope (including its boundary). The resulting
        point set and feasibility mask are stored in the grid object.

        Parameters
        ----------
        delta : float
            Grid spacing used to generate the regular sampling grid. Smaller
            values lead to denser sampling at higher computational cost.

        eps : float, default=1e-8
            Small buffer applied to the polytope geometry before feasibility checks. 
            This improves numerical robustness for points lying close to the boundary.

        Attributes Set
        --------------
        points : np.ndarray, shape (n_points, 2)
            Array containing all sampled points (grid points, boundary intersections, and polytope vertices).

        feasible_points : np.ndarray, shape (n_points,)
            Boolean mask indicating whether each sampled point lies within the boundaries of the polytope.
        """
        self.diatom._require(set_instance=True, polytope=True)

        poly = self.diatom.projection.polytope
        prepared_poly = prep(poly.buffer(eps))

        covers_xy = getattr(prepared_poly, "covers_xy", None)

        grid_points, lines = self._build_grid()
        intersect_points = self._intersection_points(lines)
        poly_vertices = np.asarray(poly.exterior.coords, dtype=float)[:-1] # type: ignore

        points = np.vstack([grid_points, intersect_points, poly_vertices])
        points = np.unique(points, axis=0)

        self.points = points

        if covers_xy is not None:
            feasible = np.fromiter(
                (covers_xy(x, y) for x, y in points), dtype=bool, count=len(points)
            )
        else:
            feasible = np.fromiter(
                (prepared_poly.covers(Point(x, y)) for x, y in points), dtype=bool, count=len(points)
                )

        self.feasible_points = feasible


    def _build_grid(self) -> tuple[NDArray[np.floating], BaseGeometry]:
        """Creates a 2D grid across the polytope bounding box.
        
        Parameters
        ----------
        delta: float
            Grid spacing used to generate the regular sampling grid. Smaller
            values lead to denser sampling at higher computational cost.

        Returns
        -------
        tuple
            Returns a tuple `(grid_points, grid_lines)`.

        grid_points: np.ndarray
            Array containing all sampled points within the bounding box. It doesn't include intersections
            between the polytope boundaries.

        grid_lines: BaseGeometry
            Geometry containing vertical and horizontal lines that define the grid sampling space.
        
        """
        delta = self.grid_delta
        poly = self.diatom.projection.polytope

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
        grid_lines = unary_union(lines)
        return grid_points, grid_lines

    