from typing import cast, Iterable

import numpy as np
from numpy.typing import NDArray
import cobra
from cobra import Model, Reaction
from cobra.util.solver import linear_reaction_coefficients

from .polytope import Projection
from .analyze import DiatomAnalyze
from .grid import DiatomGrid
from .plot import DiatomPlot
from scripts.model_io import ModelIO, load_model, file_hash
from scripts.model_clustering import ModelClustering


Numerical = int | float


class Diatom():
    """
    Class for handling a diatom metabolic model and its analysis pipeline.

    This class centralizes access to the COBRA model and orchestrates grid sampling,
    qualitative and quantitative analyses, clustering, plotting, and I/O utilities.

    Parameters
    ----------
    model_id : str
        Identifier of the metabolic model to load.
    model_name : str, default="diatom"
        Name used for outputs and saved artifacts.
    solver : str, default="gurobi"
        Linear solver backend to use with COBRApy.

    Attributes
    ----------
    model : cobra.Model
        Loaded COBRA metabolic model.
    objectives : dict[str, float]
        Mapping from reaction IDs to objective coefficients.
    non_blocked : set[str]
        Set of reaction IDs that are not blocked.
    grid : DiatomGrid
        Grid sampling and polytope discretization utilities.
    analyze : DiatomAnalyze
        Qualitative and quantitative FVA analysis utilities.
    clustering : ModelClustering
        Grid-point clustering and cluster-level analysis utilities.
    plot : DiatomPlot
        Visualization utilities.
    io : ModelIO
        Input/output utilities for saving and loading results.
    """
    def __init__(self, model_id: str, model_name: str, solver: str = "gurobi"):
        self.model_id = model_id
        self.model_name = model_name
                                 
        self.model: Model = load_model(model_id, name = model_name, solver=solver)
        self.objectives: dict[str, float] = {}
        self.non_blocked: set[str] 
        self.constraints: dict[str, tuple[Numerical, Numerical]] 

        self.projection = Projection(self)
        self.grid = DiatomGrid(self)
        self.analyze = DiatomAnalyze(self)
        self.plot = DiatomPlot(self)
        self.clustering = ModelClustering(self)
        self.io = ModelIO(self, model_name)

        self._is_sampling_instance_set: bool = False
        self.metadata: dict 
        

    def _set_objective_functions(self, objective_reactions_dict: dict[str, float] | None = None) -> None:
        """Set the objective function of the model.

        If no objective dictionary is provided, the current objective coefficients
        defined in the model are read and stored. Otherwise, the objective is replaced
        by the provided reaction–coefficient mapping.

        Parameters
        ----------
        objective_reactions_dict : dict[str, float] or None, default=None
            Mapping from reaction IDs to linear objective coefficients.
            If None or empty, the existing model objective is used.
        """
        model = self.model

        # use predefined objective functions
        if objective_reactions_dict is None or len(objective_reactions_dict) == 0:
            linear_coeffs = linear_reaction_coefficients(model)
            self.objectives = {reaction.id: coeff for reaction, coeff in linear_coeffs.items()} # should be single key,value dict, could hold more   
            return
        
        # set new objective functions
        for reaction_id, coeff in objective_reactions_dict.items():
            self.objectives[reaction_id] = coeff

        model.objective = {model.reactions.get_by_id(r): coeff for r, coeff in self.objectives.items()}  

        print(model.objective, "\n") 

    
    def _modify_bounds(self, bounds_dict: dict[str, tuple[Numerical, Numerical]]) -> None:
        """Modify reaction bounds in the model.

        Parameters
        ----------
        bounds_dict : dict[str, tuple[float, float]]
            Mapping from reaction IDs to (lower_bound, upper_bound).
        """
        for reaction_id, bounds in bounds_dict.items():
            reaction = cast(Reaction, self.model.reactions.get_by_id(reaction_id))
            reaction.bounds = bounds


    def _set_non_blocked_reactions(self) -> None:
        """Identify and store all non-blocked reactions in the model."""

        blocked = cobra.flux_analysis.find_blocked_reactions(self.model)
        reactions = cast(Iterable[Reaction], self.model.reactions)
        all_ids = [reaction.id for reaction in reactions]
        non_blocked = set(all_ids).difference(set(blocked))
        self.non_blocked = non_blocked


    def fix_growth_rates(self, mirror_model: Model, grid_point: NDArray[np.floating]) -> None:
        """Fix the biomass production rate of each community member.

        This method constrains the biomass reaction of each member to a fixed growth rate by setting its 
        lower and upper bounds to the same value.

        IMPORTANT: This method must be called *only* inside a model context manager, e.g.::

            with community_model:
                community.apply_member_fraction_bounds(community_model, fractions)
                
        Otherwise, reaction bounds will be permanently modified.

        Parameters
        ----------
        mirror_community_model : cobra.Model
            Community model whose biomass reaction bounds will be temporarily constrained.

        member_mu : np.ndarray, shape (n_members,)
            Fixed biomass production rates for each community member. Each value is imposed as an equality 
            constraint: v_biomass_i = member_mu[i]
        """

        # change bounds for each objective reaction
        for index, reaction_id in enumerate(self.analyze.analyzed_reactions): # member_objectives should be single key dictionary
            value = grid_point[index]
            reaction = cast(Reaction, mirror_model.reactions.get_by_id(reaction_id))
            reaction.bounds = (value, value)    


    def _require(
        self, 
        set_instance: bool = False,
        polytope: bool = False, 
        grid_points: bool = False, 
        clusters: bool = False, 
        qual_vector: bool = False, 
        qfca: bool = False,
    ) -> None:
        """Internal consistency check for required analysis stages.

        Raises a RuntimeError if a requested artifact has not been computed yet.
        """
        if set_instance and not self._is_sampling_instance_set:
            raise RuntimeError(f"Sampling instance hasn't been set yet. Run {self.set_sampling_instance.__name__} first!")
        
        if polytope and self.projection.polytope.is_empty:
            raise RuntimeError(f"Projected polytope not yet computed. Run {self.projection.project_polytope_2d.__name__} first!")

        if grid_points and self.grid.points.size == 0:
            raise RuntimeError(f"Grid points not yet computed. Run {self.grid.sample_polytope.__name__} first!")
        
        if qual_vector and self.analyze.qual_vector.empty:
            raise RuntimeError(f"Qualitative FVA values not yet computed. Run {self.analyze.qualitative_analysis.__name__} first!")

        if clusters and self.clustering.grid_clusters.size == 0:
            raise RuntimeError(f"Clusters not yet computed. Run {self.clustering.set_grid_clusters.__name__} first!")

        if qfca and self.analyze.qFCA.empty:
            raise RuntimeError(f"qFCA not yet computed. Run {self.analyze.quan_FCA.__name__} first!")


    def set_sampling_instance(
        self,
        constraints: dict[str, tuple[Numerical, Numerical]],
        reaction_tuple: tuple[str, str],
        grid_delta: float,
        n_clusters: int,
        save_files: bool,
        load_files: bool,
    ) -> None:
        """Configure and initialize a sampling experiment for the current diatom model.

        This method defines the biological experiment (model + constraints + objective)
        and the numerical sampling configuration (projection resolution, grid spacing,
        clustering). It also prepares the filesystem structure used to cache and reuse
        previously computed results.

        The experiment identity is determined exclusively by:
            - model file and its content hash
            - reaction bounds (constraints)
            - reaction tuple defining the 2D projection

        Numerical parameters such as `grid_delta`, `n_sampling_angles`, and
        `n_clusters` affect only how the feasible region is explored and analyzed,
        not the biological identity of the experiment.

        Parameters
        ----------
        constraints : dict[str, tuple[Numerical, Numerical]]
            Mapping from reaction IDs to (lower_bound, upper_bound).
            These bounds redefine the feasible region of the model and therefore
            determine the biological identity of the experiment.

        reaction_tuple : tuple[str, str]
            Pair of reaction IDs defining the 2D projection space.
            The first reaction is treated as the x-axis and the second as the y-axis.
            The second reaction is also set as the optimization objective.

        n_sampling_angles : int
            Number of angular directions used to approximate the 2D projected
            polytope boundary. Higher values yield a more accurate projection
            at increased computational cost.

        grid_delta : float
            Spacing between grid points used to discretize the projected polytope.
            Controls sampling resolution. Smaller values increase resolution and
            computational cost but do not change the feasible region.

        n_clusters : int
            Initial number of clusters used for grid-based clustering analysis.

        save_files : bool
            If True, all computed results (FVA results, clustering outputs, 
            plots, and dataframes) are stored to disk.

        load_files : bool
            If True, previously computed results are loaded from disk when available,
            allowing reuse of cached grid-point evaluations.

        Notes
        -----
        - Reaction bounds are rounded to 6 decimal places to ensure deterministic
        hashing and consistent floating-point representation.
        - Constraints are sorted before hashing to guarantee stable experiment IDs.
        - The experiment hash uniquely identifies the biological configuration and
        determines the root directory for result storage.
        - Numerical sampling parameters are not included in the experiment hash,
        enabling reuse of previously computed grid points across resolutions.

        Side Effects
        ------------
        - Modifies model bounds according to `constraints`.
        - Sets the objective to maximize the second reaction in `reaction_tuple`.
        - Configures sampling and clustering parameters.
        - Generates a deterministic experiment hash.
        - Prints experiment metadata and storage location.
        - Enables caching behavior depending on `save_files` and `load_files`.

        Raises
        ------
        AssertionError
            If reaction IDs do not exist in the model or parameters are invalid.
        """
        # security assertions
        assert isinstance(reaction_tuple, tuple)
        for reaction_id in reaction_tuple:
            assert reaction_id in self.model.reactions

        for reaction_id, constraint in constraints.items():
            assert reaction_id in self.model.reactions
            lb, ub = constraint
            assert isinstance(lb, Numerical) and isinstance(ub, Numerical)
            constraints[reaction_id] = (round(lb, 6), round(ub, 6))

        constraints = dict(sorted(constraints.items()))
        self.constraints = constraints

        assert isinstance(grid_delta, float) and grid_delta > 0 and grid_delta <= 1
        assert isinstance(n_clusters, int) and n_clusters > 0
        assert isinstance(save_files, bool)
        assert isinstance(load_files, bool)

        # set parameters
        self._set_objective_functions({reaction_tuple[1]: 1.0})
        self._modify_bounds(constraints)

        self.analyze.analyzed_reactions = reaction_tuple
        self.grid.grid_delta = grid_delta
        self.clustering.initial_n_clusters = n_clusters

        self.io.save_files = save_files
        self.io.load_files = load_files

        # metadata handling
        metadata = {
            "model_filename": self.model_id,
            "model_hash": file_hash(self.model_id),
            "constraints": {k: list(v) for k, v in constraints.items()}, # to ensure json compatibility
            "n_constraints": len(constraints),
            "reaction_tuple": list(reaction_tuple), # to ensure json compatibility
        }
        self.metadata = metadata

        message = f"Generated hash '{self.io.sampling_hash}' for current sampling metadata:\n"
        for key, value in metadata.items():
            message += f"{key}: {value}\n"
        
        # file management
        file_location = self.io.results_directory
        save_message = (
            f"All data generated during analysis will be stored at location {file_location}."
            if save_files else 
            "Data generated during analysis will not be stored."
        )
        load_message = (
            f"Data will be loaded from location {file_location} if available." 
            if load_files else 
            "No data will be loaded during analysis."
        )
        
        message += f"\n{save_message}\n\n{load_message}"
        print(message)

        if save_files:
            self.io.write_metadata()
        self._is_sampling_instance_set = True
        
