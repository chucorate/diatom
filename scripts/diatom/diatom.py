from typing import cast, Iterable
import numpy as np
from numpy.typing import NDArray


import cobra
from cobra import Model, Reaction
from cobra.util.solver import linear_reaction_coefficients


from .analyze import DiatomAnalyze
from .grid import DiatomGrid
from .plot import DiatomPlot
from scripts.model_io import ModelIO, load_model
from scripts.model_clustering import ModelClustering


Numerical = int | float


class Diatom():
    """Abstract base class for ecosystem models

    This class is meant to be inherited and not to be used on its own. 
    Holds all attributes and methods in common between FullEcosystem and PrecomputedEcosystem."""

    def __init__(self, model_id: str, model_name: str = "diatom"):
        self.model_id = model_id
        self.model_name = model_name
                                 
        self.model: Model = load_model(model_id, name = "diatom")
        self.objectives: dict[str, float] = {}
        self.non_blocked: set[str] = set()

        self.grid = DiatomGrid(self)
        self.analyze = DiatomAnalyze(self)
        self.plot = DiatomPlot(self)
        self.clustering = ModelClustering(self)
        self.io = ModelIO(self, model_name)
        


    def set_objective_functions(self, objective_reactions_dict: dict[str, float] | None = None) -> None:
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

        print(model.objective) 

    
    def modify_bounds(self, bounds_dict: dict[str, tuple[Numerical, Numerical]]) -> None:
        for reaction_id, bounds in bounds_dict.items():
            reaction = cast(Reaction, self.model.reactions.get_by_id(reaction_id))
            reaction.bounds = bounds


    def _set_non_blocked_reactions(self) -> None:
        """Stores all non blocked reactions."""

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


