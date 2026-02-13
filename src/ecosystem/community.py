from typing import Any, Iterable, TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

import cobra
from cobra import Model, Reaction
from cobra.util.array import create_stoichiometric_matrix
from benpy import vlpProblem
from benpy import solve as bensolve

Numerical = int | float

if TYPE_CHECKING:
    from .base import BaseEcosystem


class EcosystemCommunity():
    """
    exchange_metabolite_info : dict[str, dict[str, dict]]
        Mapping from metabolite ids to per-member exchange metadata.

    member_reactions : dict[str, list[str]]
        Mapping from member model ids to reaction ids belonging to each member."""

    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem
        self.exchange_metabolite_info: dict[str, dict[str, dict[str, Any]]] = dict()
        self.member_reactions: dict[str, list[str]] = dict()
        self.non_blocked: set[str] = set()

        self.mo_fba_sol: Any = None
        
          
    @property
    def member_model_ids(self) -> list[str]:
        return self.ecosystem.member_model_ids


    @property
    def community_model(self) -> Model:
        return self.ecosystem.community_model
    

    def set_pool_bounds(self, pool_metabolites: dict[str, tuple[Numerical, Numerical]], 
                        bioCons: Numerical | None = None) -> None:
        """
        Changes pool exchange reaction bounds for pool metabolites.

        For each metabolite in "pool_metabolites", the bounds of the corresponding pool exchange reaction
        (EX_<metabolite_id>) are updated. If a metabolite is not part of the pool a warning is raised and nothing is done.
        
        If the metabolite is consumable from the pool (lower bound <= 0) and "bioCons" is provided, the lower bound 
        of each member's exchange reaction for that metabolite is set to "bioCons", limiting individual consumption
        independently of the pool availability.

        Parameters
        ----------
        pool_metabolites: dict[str, tuple[Numerical, Numerical]]
            Dictionary mapping metabolite ids (without prefix, e.g. 'glyc_e')
            to (lower_bound, upper_bound) tuples for the pool exchange reactions.

        bioCons: Numerical or None, optional
            Lower bound imposed on member exchange reactions when the metabolite
            can be consumed from the pool. If None, no per-member consumption
            constraint is applied.
        """
        
        for metabolite_id, metabolite_bounds in pool_metabolites.items():
            if metabolite_id not in self.exchange_metabolite_info:
                print(f"Warning: {metabolite_id} is not part of pool metabolites. Skipping.")
                continue

            # Changing pool exchange reaction bounds
            reaction_id = f"EX_{metabolite_id}"
            pool_ex_reaction = cast(Reaction, self.community_model.reactions.get_by_id(reaction_id))
            pool_ex_reaction.bounds = metabolite_bounds

            # Changing members exchange reaction bounds
            exchange_metabolite = self.exchange_metabolite_info[metabolite_id]

            for member_info in exchange_metabolite.values():
                exchange_id = member_info.get('ex_id')

                if exchange_id is None:
                    continue

                member_ex_reaction = cast(Reaction, self.community_model.reactions.get_by_id(exchange_id))

                if metabolite_bounds[0] <= 0 and bioCons is not None:
                    member_ex_reaction.lower_bound = bioCons
        

        self.non_blocked = set()   


    def _get_exchange_attribute_df(self, metabolite_attribute: str) -> pd.DataFrame: 
        """Return a DataFrame with a fixed exchange attribute for each
        pool metabolite (rows) and each member model (columns). """

        assert metabolite_attribute in ['m_id', 'name', 'formula',' charge', 'ex_id', 'bounds']
    
        sorted_metabolite_indexes: list[str] = sorted(self.exchange_metabolite_info.keys()) 
        sorted_model_ids: list[str] = sorted(self.member_model_ids) 
        rows = list()

        for metabolite_index in sorted_metabolite_indexes:
            info = self.exchange_metabolite_info[metabolite_index]
            row = [info[model_id][metabolite_attribute] for model_id in sorted_model_ids]
            rows.append(row)
        
        df = pd.DataFrame(data=rows, index=sorted_metabolite_indexes, columns=sorted_model_ids)

        return df


    def set_member_exchange_bounds(self, member_model_id: str, 
                                   exchange_metabolites: dict[str, tuple[Numerical, Numerical]]) -> None:
        """Changes member exchange reaction bounds of metabolites in exchange_metabolites.
        If a metabolite is not part of the exchanges a warning is raised and nothing is done.      
    
        Parameters
        ----------
        member_model_id: str
            Model id of the community member whose exchange reactions are modified.

        exchange_metabolites: dict[str, tuple[Numerical, Numerical]]
            Dictionary mapping metabolite ids (without prefix, e.g. 'glyc_e')
            to (lower_bound, upper_bound) tuples for the exchange reactions.
        """    
        exchange_id_df = self._get_exchange_attribute_df('ex_id') # id of member exchange reactions   
        
        for metabolite, bounds in exchange_metabolites.keys():
            if metabolite not in exchange_id_df.index:
                print(f"No exchange or pool reactions for {metabolite}. Skypping")
                continue

            reaction_id = exchange_id_df.loc[metabolite, member_model_id]
            if not isinstance(reaction_id, str):
                print(f"No exchange reaction for {metabolite} in {member_model_id}. Skypping...")
                continue

            reaction = cast(Reaction, self.community_model.reactions.get_by_id(reaction_id))
            reaction.bounds = bounds
            self.exchange_metabolite_info[metabolite][member_model_id]['bounds'] = bounds
 

    # not used currently            
    def change_reaction_bounds(self, reaction_id: str, 
                               new_bounds: tuple[Numerical, Numerical]) -> tuple[Numerical, Numerical]:
        """Changes the bounds of the reaction labeled by "reaction_id". Returns old bounds."""
        community_model = self.community_model
        reaction = cast(Reaction, community_model.reactions.get_by_id(reaction_id))
        old_bounds = reaction.bounds
        reaction.bounds = new_bounds

        return old_bounds

    
    # public
    def set_member_reactions(self) -> None:
        """Builds a dictionary mapping each member model id to the list of reaction ids belonging to that model.

        Member model ids may contain underscores. Reactions are assigned to the
        member whose model id is the longest prefix of the reaction id.
        """
        member_reactions: dict[str, list[str]] = {model_id: [] for model_id in self.member_model_ids}

        for reaction in self.community_model.reactions:
            # searches if there are members associated with the reaction
            matches = [model_id for model_id in self.member_model_ids if reaction.id.startswith(model_id)]

            if not matches:
                continue

            model_id = max(matches, key=len)
            member_reactions[model_id].append(reaction.id)

        self.member_reactions = member_reactions         
        

    def _set_non_blocked_reactions(self) -> None:
        """Stores all non blocked reactions."""

        blocked = cobra.flux_analysis.find_blocked_reactions(self.community_model)
        reactions = cast(Iterable[Reaction], self.community_model.reactions)
        all_ids = [reaction.id for reaction in reactions]
        non_blocked = set(all_ids).difference(set(blocked))
        self.non_blocked = non_blocked


    def apply_member_fraction_bounds(self, mirror_community_model: Model, member_fractions: np.ndarray) -> None:
        """
        Scales reaction bounds according to community member fractions.

        This method updates reaction bounds in the provided community model by scaling each 
        member's reactions proportionally to its relative abundance in the community.

        IMPORTANT: This method must be called *only* inside a model context manager, e.g.::

            with community_model:
                community.apply_member_fraction_bounds(community_model, fractions)

        Calling this method outside a `with model:` block will permanently modify
        reaction bounds and may lead to incorrect cumulative scaling.

        Parameters
        ----------
        mirror_community_model : cobra.Model
            Community model whose reaction bounds will be temporarily modified.

        member_fractions : np.ndarray
            Relative abundance of each community member. 
        """
        # here we set fractions to zero to avoid errors setting bounds
        if np.all(np.isnan(member_fractions)): 
            member_fractions = np.array([0.0]*member_fractions.size)   

        # reactions are assign to each community member
        # community.set_member_reactions()
        for i, member in enumerate(self.member_model_ids):
            member_fraction = member_fractions[i]
            member_reactions = self.member_reactions[member]
           
            # reaction bounds are updated, accounting for members fractions in the community 
            for reaction_id in member_reactions:
                reaction = cast(Reaction, mirror_community_model.reactions.get_by_id(reaction_id))
                old_bounds = reaction.bounds
                reaction.bounds = (old_bounds[0] * member_fraction, old_bounds[1] * member_fraction)


    def fix_growth_rates(self, mirror_community_model: Model, mu_array: NDArray[np.floating]) -> None:
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
        for index, member_objectives in enumerate(self.ecosystem.objectives):    
            if len(member_objectives) != 1:
                raise RuntimeError(f"Warning: More than one reaction in {self.member_model_ids[index]} objective function. Not supported!")        
                
            #new bounds for member ix objective function reaction:
            new_bounds = (mu_array[index], mu_array[index])    
 
            #change bounds for each objective reaction
            for reaction_id in member_objectives.keys(): # member_objectives should be single key dictionary
                reaction = cast(Reaction, mirror_community_model.reactions.get_by_id(reaction_id))
                reaction.bounds = new_bounds


    # VLP PENDING ===========================================================================================


    def _to_vlp(self, **kwargs):        
        """Returns a vlp problem from EcosystemModel."""
        # We are using bensolve-2.0.1:
        # B is coefficient matrix
        # P is objective Matrix
        # a is lower bounds for B
        # b is upper bounds for B
        # l is lower bounds of variables
        # s is upper bounds of variables
        # opt_dir is direction: 1 min, -1 max
        # Y,Z and c are part of cone definition. If empty => MOLP
        
        community_model = self.community_model
        Ssigma = create_stoichiometric_matrix(community_model, array_type="lil")
        
        vlp = vlpProblem(**kwargs)
        m, n = Ssigma.shape # mets, reactions
        q = self.ecosystem.size # number of members 
        vlp.B = Ssigma
        vlp.a = np.zeros((1, m))[0]
        vlp.b = np.zeros((1, m))[0]
        vlp.l = [r.lower_bound for r in community_model.reactions] 
        vlp.s = [r.upper_bound for r in community_model.reactions] 
        
        vlp.P = lil_matrix((q, n))
        vlp.opt_dir = -1
        
        for i, member_objectives in enumerate(self.ecosystem.objectives):
            for rid, coeff in member_objectives.items():
                rindex = community_model.reactions.index(rid)
                vlp.P[i,rindex] = coeff 
                
        vlp.Y = None
        vlp.Z = None
        vlp.c = None

        return vlp  
    

    def _solve_mo_fba(self, bensolve_opts = None) -> None:
       
        if bensolve_opts is None:
            bensolve_opts = vlpProblem().default_options
            bensolve_opts['message_level'] = 0
        
        vlp_eco = self._to_vlp(options = bensolve_opts)    
        self.mo_fba_sol = bensolve(vlp_eco)


    def _get_pareto_front(self) -> np.ndarray:
        #1. Front vertex:
        vv = self.mo_fba_sol.Primal.vertex_value[np.array(self.mo_fba_sol.Primal.vertex_type)==1]
        
        n_neg_vals = np.sum(vv<0)
        if n_neg_vals > 0:
            print('warning: Negative values in Pareto Front..')
            print(vv[vv<0])
            print("Changing negative values to zero...")
            vv[vv<0] = 0   

        return vv
    
