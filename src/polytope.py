import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .constants import MAX_BRETL_ITERATIONS

if TYPE_CHECKING:
    from .metabolic_experiment import MetabolicExperiment


class Vertex:
    """Node representing a vertex of the projected polytope boundary.

    Each vertex stores its 2D coordinates and a pointer to the next vertex in the boundary 
    traversal. The `expanded` flag indicates whether the outgoing edge starting at this vertex 
    has already been processed by the expansion algorithm.
    """
    def __init__(self, p):
        self.x, self.y = p
        self.next: Vertex | None = None
        self.expanded: bool = False


class Projection():
    """Class used for 2D projection and boundary reconstruction of the feasible flux polytope.

    This class computes a two–dimensional projection of the feasible flux space
    defined by a pair of reactions. The boundary of the projected polytope is approximated using 
    an iterative directional LP strategy inspired by Bretl's algorithm.

    Parameters
    ----------
    parent_class : MetabolicExperiment
        Parent experiment providing access to the metabolic model, analysis settings,
        and reaction tuple defining the projection.

    Attributes
    ----------
    polytope : shapely.geometry.base.BaseGeometry
        Convex polygon representing the projected feasible region.
    """
    def __init__(self, parent_class: "MetabolicExperiment"):
        self.parent_class = parent_class

        self.polytope: BaseGeometry
        self._vertices: list[Vertex]


    def expand_vertex(self, vertex: Vertex, tol: float = 1e-6) -> Vertex | None:
        """Attempt to expand an edge of the current polygon.

        Given a vertex and its successor, this method computes an outward normal
        direction and solves a directional LP to determine whether a new extreme
        point exists between them. If the candidate point is colinear with the
        current edge (within tolerance), the edge is marked as fully expanded.

        Parameters
        ----------
        vertex : Vertex
            Starting vertex of the edge to be expanded.

        tol : float, default=1e-6
            Tolerance used to test colinearity of the candidate point.

        Returns
        -------
        Vertex or None
            A newly created vertex if expansion succeeds, or None if the edge
            cannot be further expanded.
        """
        v1 = vertex
        v2 = vertex.next
        if v2 is None:
            raise RuntimeError("Vertex has no succesor.")

        # get ortonormal direction
        v = np.array([v2.y - v1.y, v1.x - v2.x])
        v /= np.linalg.norm(v)

        xopt, yopt = self._solve_lp_direction(self.parent_class.analyze.analyzed_reactions, v)

        # test de colinealidad
        area = abs((xopt - v1.x)*(v1.y - v2.y) - (yopt - v1.y)*(v1.x - v2.x))

        if area < tol:
            vertex.expanded = True
            return None

        vnew = Vertex((xopt, yopt))
        vnew.next = v2
        v1.next = vnew
        v1.expanded = False
        return vnew


    def _solve_lp_direction(
        self, reaction_tuple: tuple[str, str], direction: tuple[float, float] | np.ndarray,
    ) -> tuple[float, float]:
        """Solve a directional LP to obtain a boundary point of the feasible flux space.

        Sets a linear objective defined by `direction` over two reactions and
        maximizes it to obtain an extreme point of the projected feasible region.

        Parameters
        ----------
        reaction_tuple : tuple[str, str]
            Pair of reaction IDs defining the projection axes.

        direction: tuple[float, float]
            Tuple of float numbers defining the objective direction in flux space.

        Returns
        -------
        tuple[float, float]
            Optimal flux values for the two reactions along the specified direction.

        Raises
        ------
        RuntimeError
            If the LP optimization does not converge to an optimal solution.
        """
        c0, c1 = direction
        reaction_id_0, reaction_id_1 = reaction_tuple
        
        with self.parent_class.model as model: 
            reaction_0 = model.reactions.get_by_id(reaction_id_0)
            reaction_1 = model.reactions.get_by_id(reaction_id_1)

            model.objective = {reaction_0: c0, reaction_1: c1}

            solution = model.optimize('maximize')  
            if solution.status != "optimal":
                raise RuntimeError(f"LP failed.")

            flux_0 = solution.fluxes[reaction_0.id]
            flux_1 = solution.fluxes[reaction_1.id]

            return float(flux_0), float(flux_1)
        

    def _initial_vertices(
        self, reaction_tuple: tuple[str, str], max_tries: int = 360, tol: float = 1e-6,
    ) -> None:
        """Find three non-colinear extreme points to initialize the polygon."""
        angles = np.linspace(0, 2*np.pi, max_tries, endpoint=False)
        points: list[tuple[float, float]] = []

        for theta in angles:
            direction = np.array([np.cos(theta), np.sin(theta)])
            p = self._solve_lp_direction(reaction_tuple, direction)
            if all(np.linalg.norm(np.array(p) - np.array(q)) > tol for q in points):
                points.append(p)
            if len(points) == 3:
                break

        if len(points) < 3:
            raise RuntimeError(f"Feasible region is degenerate or empty. Found points: {points}")

        v0, v1, v2 = (Vertex(p) for p in points)
        v0.next = v1
        v1.next = v2
        v2.next = v0

        self._vertices = [v0, v1, v2]
    
    
    def _iter_expand(self, max_iter: int = MAX_BRETL_ITERATIONS) -> None:
        """Iteratively expand polygon until closure.
        
        The polygon gets extended until there are no more vertices to expand, or until the
        maximum number of iterations has been reached."""
        n_iterations = 0

        vertices = self._vertices
        v = vertices[0]

        while n_iterations < max_iter:
            if v.expanded:
                v = cast(Vertex, v.next)
                if v == vertices[0]:
                    break
                continue

            vnew = self.expand_vertex(v)
            if vnew is not None:
                vertices.append(vnew)
                n_iterations += 1
            else:
                v = cast(Vertex, v.next)

        logging.debug(f"Number of iterations: {n_iterations}")
    
    
    def _ordered_vertices(self) -> list[Vertex]:
        """Return vertices ordered counterclockwise."""
        points = np.array([(v.x, v.y) for v in self._vertices])
        center = points.mean(axis=0)

        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        order = np.argsort(angles)

        return [self._vertices[i] for i in order]


    def project_polytope_2d(self, max_iter: int = MAX_BRETL_ITERATIONS) -> None:
        """Construct a 2D projection of the feasible flux polytope.

        Approximates the boundary of the feasible flux region using Bretl's polytope sampling
        algorithm, and then computes the convex hull of the resulting boundary points.

        Parameters
        ----------
        max_iter: int, default=1000
            Maximum number of iterations allowed to be used by Bretl polytope sampling algorithm.

        Attributes Set
        --------------
        polytope : BaseGeometry
            Convex hull of the projected feasible region.
        """
        self.parent_class._require(set_instance=True)

        self._initial_vertices(self.parent_class.analyze.analyzed_reactions)
        self._iter_expand(max_iter=max_iter)
        coords = [(v.x, v.y) for v in self._ordered_vertices()]

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)

        self.polytope = poly
        self.n_sampling_angles = len(self._vertices)


