from typing import TYPE_CHECKING
from functools import reduce

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from .base import BaseEcosystem


class EcosystemGrid():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem

        self.points: NDArray[np.floating] = np.array([]) # shape: (numPoints**2, 2)
        self.feasible_points: NDArray[np.bool] = np.array([])
        self.member_fractions: NDArray[np.floating] = np.array([])

        self.grid_dimensions: NDArray[np.floating] = np.array([0,0])
        self.points_per_axis: tuple[int, int] = (0,0)
        self.step: int = 0
        self.limits: tuple[NDArray, NDArray] = (np.array([]),np.array([]))
        

    @property
    def member_model_ids(self) -> list[str]:
        return self.ecosystem.member_model_ids
    

    @property
    def size(self) -> int:
        return self.ecosystem.size


    # GRID CONSTRUCTION =======================================================================

          
    def build_grid(self, numPoints: int, drop_zero: bool = False, relax_constraints: bool = False) -> None:
        """Builds a uniform abundance-growth space grid.
        
        The grid is generated from 0 to 1 at the first dimension, and from 0 to the maximum value
        the objective function at the second dimension. The point (0,0) can be dropped, and positive
        lower bounds can be temporarily relaxed to 0, in order to increase the maximum objective value.

        Assumes the model objective function is community growth, and hardcodes the grid to be
        two dimensional.

        Parameters
        ----------
        numPoints: int
            Number of points along each dimension.
        
        drop_zero: bool, optional
            If True, drops the first point of the grid. Defaults to False.

        relax_constraints: bool, optional
            If True, all positive lower bounds for each reaction are set temporarily to 0
            in order to compute the maximum objective value of the model. Defaults to False.
        """
        with self.ecosystem.community_model as community_model:
            # compute max_objective_value by relaxing constraints such as ATPm
            if relax_constraints:
                for reaction in community_model.reactions:
                    if reaction.lower_bound > 0:
                        reaction.lower_bound = 0
            
            max_objective_value = community_model.slim_optimize()

        size = self.size
        assert size == 2 # hardcoded 2D 

        mins = np.array([0.0, 0.0]) 
        maxs = np.array([1.0, max_objective_value])
        self.grid_dimensions = maxs
        print(f'Maximum community: {max_objective_value}')
        
        # builds 1D slices
        slices = [np.linspace(mins[i], maxs[i], numPoints) for i in range(size)]
        # builds 2D grid
        rgrid = np.array(np.meshgrid(slices[0], slices[1]))
        grid_points = np.column_stack([rgrid[i,:].ravel() for i in range(size)]) # shape: (numPoints**2, 2)
        print(f"points shape: {grid_points.shape}")
     
        # skips first point if true
        if drop_zero:
            grid_points = grid_points[1:]
            mins = np.asarray(np.min(grid_points, axis=0))
            maxs = np.asarray(np.max(grid_points, axis=0))
        
        self.points = grid_points
        self.points_per_axis = (numPoints, numPoints)
        self.limits = (mins, maxs)


    def set_member_fractions(self, points_range: np.ndarray | None = None):
        """Builds population fraction distributions from the grid points.

        Each grid point is assumed to be of the form (f_1, com_u), where f_1 represents
        the fraction of the first organism in a two-organism community. The fraction of
        the second organism is computed as 1 - f_1.

        Optionally, a subset of grid points can be selected using a mask or index array.

        Parameters
        ----------
        points_range : np.ndarray or None, optional
            Boolean mask or index array used to select a subset of grid points.
            If None, all grid points are used.
        """
        if self.points is None:
            raise RuntimeError('Grid points are not set yet!')
            
        points = self.points
        if points_range is not None:
            points = points[points_range]
        
        # for each point, selects the fraction of organism 1 (p[0]), and computes the fraction of organism 2
        self.member_fractions = np.array([[p[0], 1-p[0]] for p in points]) 
            

    def get_2D_slice(self, model_ids: list[str], fixed_values: list): #NJ DELETE THIS FUNCTION
        if self.size - len(model_ids) != 2:
            raise RuntimeError("Two members with non-fixed values required! No more, no less.")
        
        
        members_to_fix = range(len(model_ids))
        #valores mas cercanos en la grilla a fixed_values
        closest_values = [self._get_closest_grid_value(model_ids[i], fixed_values[i]) for i in members_to_fix]
        fixed_member_indexes = [self.member_model_ids.index(x) for x in model_ids]
        
        
        grid_points = self.points
        #grid_points_values = self.points_values # aqui va matriz de puntos x reacciones con valores cualitativos
                                         # calculados a partir de fva. 
        
        #indices de puntos en grilla donde el valor una dimension es igual a su closest_values
        filtered_indexes_list =  [np.where(grid_points[:,fixed_member_indexes[i]] == closest_values[i])[0] for i in members_to_fix]
        
        #interseccion de indices, i.e., indices slice
        slice_indexes =  reduce(np.intersect1d, filtered_indexes_list)
        #slice 2D de la grilla
        #filtered_points = grid_points[slice_indexes,:] 
        #filtered_values = grid_points_values[slice_indexes,:]
        
        free_members = list(set(self.member_model_ids).difference(set(model_ids))) # model ids of members with non-fixed objective values
        free_member_indexes = [self.member_model_ids.index(x) for x in free_members]
        #Puntos se reducen a las dimensiones de los free members:
        #slice_points = filtered_points[:,free_member_indexes]
        #slice_points_values = filtered_values[:,free_member_indexes]
          
        #return slice_points, slice_points_values , free_member_indexes
        return [slice_indexes, free_member_indexes]    
    

    def _get_closest_grid_value(self, model_id, fixed_value):
        member_index = self.member_model_ids.index(model_id)
        member_min = self.limits[0][member_index]
        member_max = self.limits[1][member_index]
        
        if fixed_value < member_min or fixed_value > member_max:
            raise RuntimeError("Value %d for %s out of range" % (fixed_value, model_id))
        
        shifted_value = fixed_value - member_min
        n_steps = shifted_value//self.step          
        
        p1 = n_steps * self.step
        p2 = (n_steps + 1) * self.step  
        
        if (shifted_value - p1) < (p2 - shifted_value):
            closest_value = member_min + p1
        else:
            closest_value = member_min + p2        
        
        return closest_value    


    def resolve_2D_slice(self, model_ids, fixed_values):
        # get full 2D slice:   
        if self.size == 2:
            if len(model_ids) > 0:
                print("Only two members in community!!") 
                print("Full grid will be plotted and fixed values for %s will be ignored..." % str(model_ids))
            free_member_model_ids = self.member_model_ids
            free_member_indexes = [0,1]
            full_slice_indexes = np.arange(len(self.points)) 
            
        else:              
            full_slice_indexes, free_member_indexes = self.get_2D_slice(model_ids, fixed_values)
            free_member_model_ids = [self.member_model_ids[x] for x in free_member_indexes]

        return full_slice_indexes, free_member_indexes, free_member_model_ids


    def get_slice_points(self, full_slice_indexes, free_member_indexes):

        if self.feasible_points.size == 0: 
            slice_indexes    = full_slice_indexes
            slice_points     = self.points[full_slice_indexes,:][:,free_member_indexes] 
            slice_pfractions = self.member_fractions[full_slice_indexes]

            return slice_points, slice_pfractions, slice_indexes

        feasible_indexes    = np.where(self.feasible_points)[0]
        feasible_points     = self.points[feasible_indexes]
        feasible_pfractions = self.member_fractions[feasible_indexes]
                
        slice_indexes    = np.isin(feasible_indexes, full_slice_indexes)
        slice_points     = feasible_points[slice_indexes,:][:,free_member_indexes]  
        slice_pfractions = feasible_pfractions
            
        return slice_points, slice_pfractions, slice_indexes
    

    def _calculate_community_growth(self, feasible=False): #NJ DELETE THIS FUNCTION
        cgrowth = np.sum(self.points,axis=1) 
        if feasible:
            if self.feasible_points.size == 0:
                print('feasible points have not been previously established! Returning values for all points')
            else:
                cgrowth = cgrowth[self.feasible_points]
                       
        return cgrowth     


    """

    def get_polytope_vertex(self, expand: bool = True):
  
        """"""
        polytope: pareto front + axes segments + extra segments perpendicular to axes dimensions where 
        pareto solutions don't reach 0 values. 
        (assumption: objective functions can only take positive values)
        """"""
        pareto_front = self.ecosystem.community._get_pareto_front()

        #2. origin
        ov = np.zeros((1, self.size))
        
        if expand == True:
            #3. Extra points that close polytope (only if points (0,0,0,...,xi_max,0,...0) are not pareto front 
            # points but they are feasible) 
            
            #MP: si hay que incluir estos puntos significa que hay miembros que son givers: i.e. pueden crecer
            # a su máxima tasa y aguantar que otros miembros crezcan también
            # si un punto (0,0,0,...,xi_max,0,...0) no es factible entonces el miembro i necesita que otros crezcan 
            # para poder crecer (needy).
            
            #3.1 Check if points  (0,0,0,...,xi_max,0,...0) are part of the pareto front
            n = self.size - 1
            all_zeros_but_one = np.argwhere(np.sum(pareto_front == 0,axis=1)==n) # index of (0,0,...,xi_max,0,0...0) points
            all_zeros_but_one = all_zeros_but_one.flatten()
    
            # indexes i of non-zero member in (0,0,...,xi_max,0,0...0) pareto points, 
            # i.e. members that are not givers nor needy. 
            non_zero_dims =  np.argmax(pareto_front[all_zeros_but_one,:], axis = 1) 
            
            # givers and/or needy members:
            givers_or_needy_indexes = np.setdiff1d(np.array(range(self.size)), non_zero_dims) 
            gn_total= len(givers_or_needy_indexes)    
        
            #3.2 Check if non-pareto points (0,0,0,...,xi_max,0,...0) are feasible 
            if gn_total >0:
                # max values for giver_or_needy members:
                max_vals = np.max(pareto_front, axis=0)
                cpoints = np.diag(max_vals)
                to_check = cpoints[givers_or_needy_indexes,:]
                
                are_feasible = self.ecosystem.analyze._feasibility_analysis(to_check)
                
                ev = to_check[are_feasible, :] 

                polytope_vertex = np.concatenate((pareto_front,ov,ev), axis=0)
            else: 
                polytope_vertex = np.concatenate((pareto_front,ov), axis=0)
        
        else:
                polytope_vertex = np.concatenate((pareto_front,ov), axis=0)

        return polytope_vertex 
    
    """
    
