import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cobra.flux_analysis import flux_variability_analysis
from cobra.util.array import create_stoichiometric_matrix
from cobra.io.mat import _cell
from cobra import Reaction

import copy
import scipy.io as sio
from collections import OrderedDict
from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem


qualitative_dict = {
    -3.0: '-', 
    -2.0: '--',
    -1.0: '-0',
    0.0: '0',
    1.0: '0+',
    2.0: '++',
    3.0: '+',
    4.0: '-+',
    5.0: 'err',
    100.0: 'var'
    }


class EcosystemAnalyze():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem  # parent class
        self.qual_vector_df = None # public
        self.rxn2cluster: list[str] | None = None # public
        self.fva_results = None
        self.qFCA = None
        self.qualitative_dict: dict[float, str] = qualitative_dict

        self.coupled_rxns: dict[str, dict] = {}      
        self.member_blocked: dict[str, list]  = {}


    @property
    def member_model_ids(self):
        return self.ecosystem.member_model_ids


    # CUALITATIVE ANALYSIS

    
    def select_reactions_for_clustering(self) -> None:
        """Selects the set of community reactions to be used for FVA and clustering analyses.

        This method determines which reactions of the community model should be
        considered for analysis based on the availability of FCA results for all
        member models.

        If FCA results are missing for at least one member model, all non-blocked community 
        reactions are selected.

        If FCA results are available for all member models:
        - Blocked reactions in any member are excluded.
        - For each fully coupled reaction set, only one representative reaction is selected.
        - Reactions not accounted for by member models (e.g. pool or exchange reactions) are also included.
        
        Notes
        -----
        - This method assumes that FCA results, when available, have been previously
        stored using `store_fca_results`.
        - Pool bounds and community structure must be set before calling this method.
        - The method does not modify the community model; it only selects a subset
        of reactions for analysis.
        """
        community = self.ecosystem.community
        missing_models = set(self.member_model_ids) - set(self.coupled_rxns.keys())
    
        if missing_models: # FCA results are incomplete. Non-blocked reactions are obtained and ALL those reactions are used.        
            print(f"Missing FCA results for: {missing_models}.\nUsing non-blocked reactions only.")
            community._set_non_blocked_reactions()  

            rxn2cluster = list(community.non_blocked)  
            rxn2cluster.sort()
            self.rxn2cluster = rxn2cluster

            print("Total reactions considered for fva and clustering: %d" % len(self.rxn2cluster))
            return

        # Otherwise, FCA results are available for all members, reactions for fva and clustering are reduced accordingly.
        accounted = []      
        coupled_rep = []
              
        # blocked reactions are not considered for fva and clustering   
        for blocked in self.member_blocked.values():
            accounted += blocked
              
        # Only one reaction from each fully coupled set is used for fva and clustering      
        for coupled_sets_dict in self.coupled_rxns.values():
            # reaction representative for coupled set:   
            coupled_rep += list(coupled_sets_dict.keys())
            coupled_sets = list(coupled_sets_dict.values())
            accounted += [reaction for cset in coupled_sets for reaction in cset]
        
        # Reactions for fva and clustering are those representing coupled sets and those not in any member (pool reactions)                   
        all_reaction_ids = [reaction.id for reaction in self.ecosystem.community_model.reactions]
        missing_reactions = set(all_reaction_ids).difference(set(accounted))
        
        rxn2cluster = list(missing_reactions) + coupled_rep       
        rxn2cluster.sort()
        self.rxn2cluster = rxn2cluster

        print("Total reactions considered for fva and clustering: %d" % len(self.rxn2cluster))


    def analyze_grid(self, analysis: str = 'feasibility', update_bounds: bool = True, **kwargs) -> None:
        """
        Run an analysis over the full ecosystem grid.

        Updates attributes of `self.ecosystem.grid` and/or:
            - self.ecosystem.grid.feasible_points
            - self.qual_vector_df
            - self.fva_results

        Parameters
        ----------
        analysis : {'feasibility', 'qual_fva'}, default='feasibility'
            Type of analysis to run on full grid.
            - 'feasibility': checks if each grid point is feasible considering member fractions.
            - 'qual_fva'   : computes qualitative flux vectors and FVA results. If feasible 
            points are stored, analysis is run on those points only. FVA results are also stored.  

        update_bounds : bool, default=True
            Whether reaction bounds should be updated using member fractions
            at each grid point.

        **kwargs :
            Additional keyword arguments passed to the underlying analysis
            routines.
        """
        # calculate member fractions for each grid point if they are not stored
        if self.ecosystem.grid.member_fractions.size == 0:
            self.ecosystem.grid.set_member_fractions()
        
        # run analysis
        if analysis == 'feasibility':
            self._feasibility_analysis(update_bounds=update_bounds, **kwargs)
            
        elif analysis == 'qual_fva':
            self._qualitative_analysys(update_bounds=update_bounds, **kwargs)   

        else:
            raise ValueError(f"Non valid analysis option: {analysis}") 

           
    def _feasibility_analysis(self, update_bounds: bool = False, **kwargs) -> None:
        #step 2: run feasibility analysis for all grid points  
        point_array            = self.ecosystem.grid.points               
        member_fractions_array = self.ecosystem.grid.member_fractions
        n_points               = point_array.shape[0]
        n_frac                 = member_fractions_array.shape[0]

        if not update_bounds:
            feasibility_mask = [self._analyze_point(grid_point, None, analysis='feasibility', 
                                                    update_bounds=False, **kwargs) for grid_point in point_array] 
            
        else:
            if member_fractions_array.size == 0 or n_points != n_frac:
                raise RuntimeError("Missing or incomplete member fractions array. Cannot update reaction bounds!") 
              
            feasibility_mask = [self._analyze_point(point_array[i], member_fractions_array[i], 
                                                    analysis='feasibility', update_bounds=True, **kwargs)
                                                    for i in range(n_points)] 
            
        self.ecosystem.grid.feasible_points = np.array(feasibility_mask)
        
        n_feasible = sum(self.ecosystem.grid.feasible_points)
        print(f"grid feasible points: {n_feasible}/{n_points}")


    def _qualitative_analysys(self, update_bounds: bool = False, **kwargs) -> None:
        point_array            = self.ecosystem.grid.points               
        member_fractions_array = self.ecosystem.grid.member_fractions
        feasible_points        = self.ecosystem.grid.feasible_points

        if feasible_points is None:
            print("Warning: Feasible points have not been calculated. Running qualitative fva over full grid")
            df_index = np.arange(point_array.shape[0])
        else:
            print("Running qualitative fva over grid feasible points...")
            point_array = point_array[feasible_points,:]    
            member_fractions_array = member_fractions_array[feasible_points,:]     
            df_index =  np.where(feasible_points)[0]
        
        rtuples = self.calculate_qual_vectors(point_array,member_fractions_array,update_bounds=update_bounds, **kwargs)
            
        qual_vector_list, fva_results =  map(list, zip(*rtuples))    
        self.qual_vector_df = pd.DataFrame(np.array(qual_vector_list),columns = self.rxn2cluster, index=df_index)
            
        fva_results = np.dstack(fva_results)
        fva_results = np.rollaxis(fva_results,-1)
            
        self.fva_results = fva_results  


    def calculate_qual_vectors(self,point_array, pfraction_array=None, update_bounds=False, **kwargs):
        
        # Check for reactions selected for FVA and clustering
        if self.rxn2cluster is None:
            print("No reactions previously selected for FVA and clustering!\nSetting reactions to cluster...")
            self.select_reactions_for_clustering()        
        
        if not update_bounds:
            print("Warning:Calculating qualitative vectors without updating reaction bounds!")
            r = [self._analyze_point((p,None), analysis = 'qual_fva', update_bounds=False, **kwargs) for p in point_array] 
        else:
            npoints = point_array.shape[0]
            nfrac   = pfraction_array.shape[0]
            
            if pfraction_array is None or npoints != nfrac:
                raise RuntimeError("Missing or incomplete member fractions array. Cannot update rxn bounds!!") 
            else:
                coord_frac = [(point_array[ix],pfraction_array[ix]) for ix in range(npoints)]  
                #list of tuples
                r = [self._analyze_point(x, analysis = 'qual_fva', update_bounds = True, **kwargs) for x in coord_frac] 
         
        return r
    

    # ================================================================== ANALYZE POINT ==================================================================


    @staticmethod
    def _qual_translate(interval: pd.DataFrame, delta: float = 1e-4) -> float:
        """
        Translate FVA min/max values into qualitative states. Outputs the numeric value that maps
        to the qualitative state in `self.qualitative_dict`.
        """
        fmax, fmin = float(interval.maximum.item()), float(interval.minimum.item())

        same_value = abs(fmax - fmin) < delta
        pos_max = fmax > delta
        neg_max = fmax < -delta
        pos_min = fmin > delta
        neg_min = fmin < -delta
        zero_max = abs(fmax) <= delta
        zero_min = abs(fmin) <= delta

        rules = [
            (lambda: neg_min and neg_max and same_value, -3.0),
            (lambda: neg_min and neg_max, -2.0),
            (lambda: neg_min and zero_max, -1.0),
            (lambda: zero_min and zero_max, 0.0),
            (lambda: zero_min and pos_max, 1.0),
            (lambda: pos_min and pos_max, 2.0),
            (lambda: pos_min and pos_max and same_value, 3.0),
            (lambda: neg_min and pos_max, 4.0)
        ]

        for rule, qualitative_value in rules:
            if rule():
                return qualitative_value

        # no qualitative state could be determined. Check qualitative_dict for conversions
        return 5.0
        

    def _analyze_point(self, grid_point: np.ndarray, member_fractions: np.ndarray | None, 
                       analysis: str = 'feasibility', update_bounds: bool = False, delta: float = 1e-9) -> bool | tuple:
        """Analyze a single grid point of the ecosystem parameter space.

        This method evaluates either the feasibility of a grid point or computes
        qualitative flux variability information, optionally updating reaction
        bounds according to community member fractions.

        Parameters
        ----------
        grid_point : np.ndarray, shape (2,)
            Grid coordinates defining the community state. The first coordinate corresponds to the 
            fraction of total biomass growth assigned to member 1, and the second coordinate corresponds 
            to the total community biomass production rate.

            Internally, this is transformed into per-member biomass growth rates:
                mu_1 = fraction * mu_total
                mu_2 = (1 - fraction) * mu_total

        member_fractions : np.ndarray or None, shape (n_members,)
            Relative abundance of each community member. These fractions are used to scale reaction 
            bounds when `update_bounds=True`. If all entries are NaN, they are interpreted as zero 
            to avoid invalid bound updates.

            This argument is ignored when `update_bounds=False`.

        analysis : {"feasibility", "qual_fva"}, default="feasibility"
            Type of analysis to perform:
            - "feasibility": checks whether the grid point admits a feasible solution.
            - "qual_fva": computes qualitative flux variability categories for a selected set of reactions.

        update_bounds : bool, default=False
            If True, reaction bounds are scaled according to member fractions before running the analysis. 
            Bound updates are temporary and must be executed inside the model context manager.

        delta : float, default=1e-9
            Numerical tolerance used when translating flux variability ranges into
            qualitative categories. Only relevant for `analysis="qual_fva"`.

        Returns
        -------
        bool
            If `analysis="feasibility"`, returns True if the grid point is feasible,
            False otherwise.

        tuple
            If `analysis="qual_fva"`, returns a tuple `(qual_vector, fva_values)`:
            - qual_vector : list[int]
                Qualitative category assigned to each reaction.
            - fva_values : np.ndarray, shape (n_reactions, 2)
                Minimum and maximum flux values obtained from FVA.

        Notes
        -----
        All model mutations (reaction bound updates and objective fixing) are performed inside a 
        `with community_model:` context. This guarantees that all changes are reverted after the 
        analysis is completed.
        """
        # ptuple: tuple of two arrays:   (0) Grid point coordinates; 
        #                                (1) grid point member fractions; grid point
        # analysis: type of analysis to run. Options:
        #           'feasibility': check if grid point is feasible
        #           'qual_fva'   : get grid point vector of rxn qualitative values. 
        #                           
                    
        # update_bounds: if True update reaction bounds considering member community fractions 
        #                before analysis 
        # delta: threshold value to consider flux differences as zero when comparing fva min and max values
        #        ('qual_fva' option only)  
        # Returns  
        #  boolean indicating point feasibility ('feasibility' analysis) or  ('qual_val' analysis)
        #  a tuple where the first element is a list of rxns qualitative values for the analyzed grid point
        #  and the second is an array with the corresponding fva results
        community_model = self.ecosystem.community_model

        print(f"point: {grid_point}, frac: {member_fractions}")

        fraction, mu_total = grid_point
        member_mu = [fraction*mu_total, (1-fraction)*mu_total] 

        
        with community_model:
            # update member reactions bounds if required:
            if update_bounds: 
                if not isinstance(member_fractions, np.ndarray):
                    raise TypeError("member_fractions must be a numpy array")
                
                print('updating reaction bounds ...')

                # here we set fractions to zero to avoid errors setting bounds
                if np.all(np.isnan(member_fractions)): 
                    member_fractions = np.array([0.0]*member_fractions.size)                
                    
                self.ecosystem.community.apply_member_fraction_bounds(community_model, member_fractions)


            # fix member objectives to grid point value:
            for index, member_objectives in enumerate(self.ecosystem.objectives):    
                if len(member_objectives) != 1:
                    raise RuntimeError("Warning: More than one reaction in %s objective function. Not supported!!" 
                      % self.member_model_ids[index])        
                #new bounds for member ix objective function reaction:
                new_bounds = (member_mu[index], member_mu[index])    
                #newGrid NJ
                #print(point[ix])

            
                #change bounds for each objective reaction
                for reaction_id in member_objectives.keys(): # member_objectives should be single key dictionary
                    rxn = community_model.reactions.get_by_id(reaction_id)
                    rxn.bounds = new_bounds

                    #set one of objective reactions as community objective
                    #cmodel.objective = rid #commented since the model comes with an objective
            
           
            if analysis == 'feasibility': 
                #check point feasibility
                error_value = -1000
                ob = community_model.slim_optimize(error_value = error_value)  
                if ob == error_value:
                    out = False
                    print('unfeasible point')
                else:
                    out = True

                return out
        
            elif analysis == 'qual_fva':  # here we assume the point is feasible      

                if self.rxn2cluster is None:
                    raise RuntimeError('No reactions selected for fva and clustering!')
                    
                print("running FVA on grid point...")
                print((grid_point, member_fractions))
                
                rxn_fva = flux_variability_analysis(community_model, reaction_list= self.rxn2cluster)
                
                rxn_fva = rxn_fva.loc[self.rxn2cluster,:] # just to make sure reactions are in the 
                                                         # same order as rxn2cluster
                    
                print("translating to qualitative vector..")
                out = (list(rxn_fva.apply(self._qual_translate, axis = 1, delta = delta)), #qual_vector 
                             rxn_fva.values) # array with FVA results  
                
                return out 

            else:
                raise ValueError(f"Non valid analysis option: {analysis}")     
        

    # QUANTITATIVE GRID ANALYSIS


    def quan_FCA(self, grid_x, grid_y, rxns_analysis):
        #Performs quantitative Flux Coupling Analysis on two reactions (rxns_analysis) and on points of a sub-grid defined by points grid_x, grid_y
        #returns: a dataframe with columns to plot afterwards
        #Columns: flux_rxns_analysis[0], flux_rxn_analysis[1], FVA (str: minimum or maximum), point (coordinates of point)

        feasible_points = self.ecosystem.grid.points[self.ecosystem.grid.feasible_points]
        analyze_points = []
        print('Quantitative Flux Coupling analysis \n Initializing grid...')


        # a lo mejor conviene definirlo fuera
        def fraction_to_normalize(point_fractions, reaction):
            #from point_fraction computes which element of this array should be used for normalization
            #reaction: string reaction id
            fraction = ''
            for i, pre in enumerate(self.member_model_ids):
                if reaction.startswith(pre+'_'):
                    fraction = point_fractions[i]

            
            if fraction=='':
                print('No org detected, asumming community reaction')
                fraction =1
        
            return(fraction)
        
        #Match points defined by the user in grid_x, grid_y to specific points on the grid
        for y in grid_y:
            for x in grid_x:
                search_point = [x, y]
                distances = np.linalg.norm(feasible_points-search_point, axis=1)
                min_index = np.argmin(distances)
                analyze_points.append(min_index)
                print(f"the closest point to {search_point} is {feasible_points[min_index]}, at a distance of {distances[min_index]}")


        maxmin_data = []
        for this_point in analyze_points:
            eco = self.ecosystem
            community_model = copy.deepcopy(eco.community_model)
        
            this_point_coords = feasible_points[this_point]
            print('Selected point'+str(this_point_coords))
            print('This point coords '+str(this_point_coords))
            this_point_frac = [this_point_coords[0], 1-this_point_coords[0]]
            print('This point frac '+str(this_point_frac))
            point = [this_point_coords[0]*this_point_coords[1], (1-this_point_coords[0])*this_point_coords[1]] #equivalent to old grid
            print('Old grid point '+str(point))

            #update bounds
            for i, member in enumerate(self.member_model_ids):
                mfrac = this_point_frac[i]
                mrxns = eco.community.member_reactions[member]

                for rid in mrxns:
                    r = cast(Reaction, community_model.reactions.get_by_id(rid))
                    old_bounds = r.bounds
                    r.bounds = (old_bounds[0]*mfrac, old_bounds[1]*mfrac)

            for ix, member_objectives in enumerate(eco.objectives):
                new_bounds = (point[ix], point[ix])

                for rid in member_objectives.keys():
                    rxn = community_model.reactions.get_by_id(rid)
                    rxn.bounds = new_bounds

            #try:
            #define limits reactions based on theoretical max-min defined from model
            rxn_ref_fva = flux_variability_analysis(community_model, reaction_list = rxns_analysis[0])

            #define range reactions
            values_rxn_ref = np.linspace(rxn_ref_fva['minimum'].iloc[0], rxn_ref_fva['maximum'].iloc[0], num=50)

            with community_model:
                for val in values_rxn_ref:
                    rxn = community_model.reactions.get_by_id(rxns_analysis[0])
                    rxn.bounds = (val,val)
                    #compute max min
                    fva = flux_variability_analysis(community_model, reaction_list = rxns_analysis[1])
                    for i, el in enumerate(fva):
                        row_dict = dict()
                        row_dict[rxns_analysis[0]] = val/fraction_to_normalize(this_point_frac, rxns_analysis[0])
                        row_dict[rxns_analysis[1]] = fva[el].iloc[0]/fraction_to_normalize(this_point_frac, rxns_analysis[1])
                        row_dict['FVA'] = el
                        row_dict['point'] = str([round(this_point_coords[0],3), round(this_point_coords[1],3)])
                        maxmin_data.append(row_dict)

            #except:
            #    print('\n Issues with '+str(this_point_coords)+' unfeasible?')
        
        self.qFCA = pd.DataFrame(maxmin_data)
        
        
    def write_fca_input(self,model_id,file_dir, discard_zero_bound=True):
     
        file_name = "%s/%s_fca_input.mat" % (file_dir,model_id)
        community_model = self.ecosystem.community_model 
            
        rxns = community_model.reactions
        mets = community_model.metabolites
        stoich_mat = create_stoichiometric_matrix(community_model)
        rids = np.array(rxns.list_attr("id"))
        mids = np.array(mets.list_attr("id"))
        rev  = np.array(rxns.list_attr("reversibility"))*1
    
        #discard reactions from pool and other members and also reactions from prefix with zero lower and upper bounds 
        #these last reactions are added to blocked.
        
        to_discard = []
        blocked = [] 
                
        for ix in range(len(rids)):
            rid = rids[ix]
            if not rid.startswith(model_id):
                to_discard.append(ix)
            else:        
                if discard_zero_bound:            
                    r = community_model.reactions.get_by_id(rid)
                    if r.lower_bound == 0 and r.upper_bound == 0 :
                        to_discard.append(ix)
                        blocked.append(rid)     
        
        if len(to_discard)>0:
            rids = np.delete(rids,to_discard)
            rev = np.delete(rev,to_discard)
            stoich_mat = np.delete(stoich_mat, to_discard, axis=1)
                  
        
        #discard metabolites from pool and other members:
        to_discard = []
        for ix in range(len(mids)):
            mid=mids[ix]
            if not mid.startswith(model_id):
                to_discard.append(ix)
        
        if len(to_discard)>0:  
            mids = np.delete(mids,to_discard)
            stoich_mat = np.delete(stoich_mat, to_discard, axis=0)    
        
        rids =list(rids)
        mids =list(mids)
        
        #create mat objects for FCA and stored them in an output file 
        mat  = OrderedDict()    
        mat["Metabolites"] = _cell(mids)
        mat["Reactions"] = _cell(rids)
        mat["stoichiometricMatrix"] = stoich_mat
        mat["reversibilityVector"] = rev 
        
        varname1 = 'fca_input'
        varname2 = 'bound_blocked'
        #varname2 = "%s_bound_blocked" % prefix
        #sio.savemat(file_name, {varname1: mat, varname2:mat2}, oned_as="column")
        sio.savemat(file_name, {varname1: mat, varname2:_cell(blocked)}, oned_as="column")
        print("%s FCA's input in %s" % (model_id, file_name))
        print("   stoichiometric matrix : %s" % str(stoich_mat.shape))
        

    def store_fca_results(self,model_id,fca_file):

        mat_contents = sio.loadmat(fca_file)
        fctable = mat_contents['fctable']
        blocked = mat_contents['blocked'][0]
        rxn_ids = np.array([rid[0][0] for rid in mat_contents['rxns']])
        bound_blocked = np.array([rid[0][0] for rid in mat_contents['bound_blocked']])
        blocked_ids = rxn_ids[blocked==1]
        blocked_ids = list(blocked_ids) + list(bound_blocked)
        non_blocked_ids = list(rxn_ids[blocked!=1])
        
        g = np.unique(fctable==1,axis=0)
        df = pd.DataFrame(data=g, columns= non_blocked_ids)
        coupled_sets=dict() #key = id of one reaction of the coupled set (first in alpha order); 
                            #value = list of coupled rxns ids
        for x in df.index:
            coupled = list(df.columns[df.loc[x]])
            coupled.sort()
            coupled_sets[coupled[0]] = coupled   
                
        self.coupled_rxns[model_id] = coupled_sets
        self.member_blocked[model_id]= blocked_ids
        total_rxns = len(rxn_ids)+len(bound_blocked)
        print("Flux coupling results for member %s stored:" % model_id)
        print("   Total reactions: %d" % total_rxns)
        print("   Fully coupled reaction sets: %d" % len(coupled_sets))
        print("   Blocked reactions: %d" % len(blocked_ids))      
        print("-")
