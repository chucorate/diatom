from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from cobra import Reaction
from cobra.flux_analysis import flux_variability_analysis
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from .diatom import Diatom


CATEGORY_DICT = {
    -3.0: '-', 
    -2.0: '--',
    -1.0: '-0',
    0.0: '0',
    1.0: '0+',
    2.0: '+',
    3.0: '++',
    4.0: '-+',
    5.0: 'err',
    100.0: 'var'
    }


def qual_translate(fmin: np.ndarray, fmax: np.ndarray, delta: float = 1e-4) -> np.ndarray:
    """
    Translate FVA min/max values into qualitative states. Outputs the numeric value that maps
    to the qualitative state in `self.qualitative_dict`.
    """

    same_value = np.abs(fmax - fmin) < delta
    pos_max = fmax > delta
    neg_max = fmax < -delta
    pos_min = fmin > delta
    neg_min = fmin < -delta
    zero_max = np.abs(fmax) <= delta
    zero_min = np.abs(fmin) <= delta

    conditions = [
        neg_min & neg_max & same_value,
        neg_min & neg_max,
        neg_min & zero_max,
        zero_min & zero_max, 
        zero_min & pos_max,
        pos_min & pos_max & same_value,  # order here is VERY IMPORTANT, will fix later
        pos_min & pos_max, 
        neg_min & pos_max, 
    ]

    choices = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

    return np.select(conditions, choices, default=5.0)


class DiatomAnalyze():
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom
        self.polytope: BaseGeometry = Polygon()

        self.analyzed_reactions: tuple[str, str] = ("","")
        self.fva_reactions: list[str]            = []                  
        self.fva_results: np.ndarray             = np.empty((0, 0, 2)) # shape: (n_points, n_reactions, 2)

        self.qual_vector_df: pd.DataFrame        = pd.DataFrame()     
        self.category_dict: dict[float, str]     = CATEGORY_DICT
        self._empty_qual_vector: list[float] | None = None
        self._empty_fva_result: np.ndarray | None = None

        self.qFCA: pd.DataFrame = pd.DataFrame()


    def _solve_lp_direction(self, reaction_tuple: tuple[str, str], theta: float) -> tuple[float, float]:
        c0, c1 = np.cos(theta), np.sin(theta)  
        reaction_id_0, reaction_id_1 = reaction_tuple
        
        with self.diatom.model as model: 
            reaction_0 = model.reactions.get_by_id(reaction_id_0)
            reaction_1 = model.reactions.get_by_id(reaction_id_1)

            model.objective = {reaction_0: c0, reaction_1: c1}

            solution = model.optimize('maximize')  
            if solution.status != "optimal":
                raise RuntimeError(f"projection failed at theta = {theta}")

            flux_0 = solution.fluxes[reaction_0.id]
            flux_1 = solution.fluxes[reaction_1.id]

            return float(flux_0), float(flux_1)


    def project_polytope_2d(self, reaction_tuple: tuple[str, str], n_angles: int = 360) -> None:
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        boundary_points = [self._solve_lp_direction(reaction_tuple, theta) for theta in angles]

        boundary_points = np.unique(boundary_points, axis = 0)
        poly = Polygon(boundary_points).buffer(0)

        self.polytope = poly.convex_hull
        self.analyzed_reactions = reaction_tuple


    def qualitative_analysis(self, 
                             x_limits: tuple[float, float] = (-np.inf, np.inf), 
                             y_limits: tuple[float, float] = (-np.inf, np.inf),
                             only_load: bool = False, 
                             **kwargs) -> None:
        """Run qualitative FVA analysis for all (or feasible) grid points.

        Generates qualitative vectors and FVA results for each grid point and stores them
        in `self.qual_vector_df` and `self.fva_results`. If feasible points have been
        calculated, the analysis is restricted to them, otherwise the full grid is used.

        - qual_vector_df : pd.DataFrame
            Dataframe containing qualitative categories assigned to each reaction.
        - fva_results : np.ndarray, shape (n_reactions, 2)
            Minimum and maximum flux values obtained from FVA.
        """
        points           = self.diatom.grid.points               
        feasible_points  = self.diatom.grid.feasible_points

        if feasible_points.size == 0:
            print("Warning: Feasible points have not been calculated. Running qualitative fva over full grid")
            df_index = np.arange(points.shape[0])
        else:
            print("Running qualitative fva over grid feasible points...")
            filtered = (points[:, 0] > x_limits[0]) & (points[:, 0] < x_limits[1]) & (points[:, 1] > y_limits[0]) & (points[:, 1] < y_limits[1])
            analyzed_points = filtered & feasible_points

            points = points[analyzed_points, :]    
            df_index = np.where(analyzed_points)[0]

            self.diatom.grid.analyzed_points = analyzed_points
        
        fva_tuples = self._calculate_qual_vectors(points, only_load = only_load, **kwargs)
            
        qual_vector_list, fva_results = map(list, zip(*fva_tuples))    
        self.qual_vector_df = pd.DataFrame(np.array(qual_vector_list), columns=self.fva_reactions, index=df_index)
        #self.diatom.io.save_qual_df()    

        fva_results = np.dstack(fva_results)
        fva_results = np.rollaxis(fva_results, -1)
        
        print("Done!\n")
        self.fva_results = fva_results 


    def _calculate_qual_vectors(self, grid_points: np.ndarray, only_load: bool, **kwargs) -> list[tuple]:
        """Calculate qualitative FVA vectors for a set of grid points.

        Iterates over grid points and calculates qualitative FVA vectors using. Each element in the returned 
        list is a tuple `(qual_vector, fva_result)` for a point.
        """
        # Check for reactions selected for FVA and clustering
        if not self.fva_reactions:
            print("No reactions previously selected for FVA and clustering!\nSetting reactions for analysis...\n")
            
            self.diatom._set_non_blocked_reactions()  
            fva_reactions = list(self.diatom.non_blocked)  
            fva_reactions.sort()

            self.fva_reactions = fva_reactions    
        
        n_points = grid_points.shape[0]

        print("Analyzing point feasibility....")
        if only_load:
            fva_tuples = []
            for grid_point in tqdm(grid_points, total=n_points):
                loaded = self._load_if_stored(grid_point)
                if loaded is not None:
                    fva_tuples.append(loaded)
        else:
            fva_tuples = [self._analyze_point(grid_point, **kwargs) for grid_point in tqdm(grid_points, total = n_points)] 

        return fva_tuples
    
    
    def _load_if_stored(self, grid_point: np.ndarray):
        loaded = self.diatom.io.load_point(grid_point, "qual_fva")

        if isinstance(loaded, tuple):
            return loaded

        # --- fallback: placeholder ---
        if self._empty_qual_vector is None or self._empty_fva_result is None:
            n_rxns = len(self.fva_reactions)
            self._empty_qual_vector = [np.nan] * n_rxns
            self._empty_fva_result = np.full((n_rxns, 2), np.nan)

        return (self._empty_qual_vector, self._empty_fva_result)


    def _analyze_point(self, grid_point: np.ndarray, delta: float = 1e-9, **kwargs) -> tuple:
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

        analysis : {"feasibility", "qual_fva"}, default="feasibility"
            Type of analysis to perform:
            - "feasibility": checks whether the grid point admits a feasible solution.
            - "qual_fva": computes qualitative flux variability categories for a selected set of reactions.

        delta : float, default=1e-9
            Numerical tolerance used when translating flux variability ranges into
            qualitative categories. Only relevant for `analysis="qual_fva"`.

        Returns
        -------
        tuple
            Returns a tuple `(qual_vector, fva_values)`.
            
        Notes
        -----
        All model mutations (reaction bound updates and objective fixing) are performed inside a 
        `with community_model:` context manager. This guarantees that all changes are reverted after the 
        analysis is completed.
        """
        
        
        loaded_point = self.diatom.io.load_point(grid_point, "qual_fva")
        if isinstance(loaded_point, tuple):
            return loaded_point
        

        with self.diatom.model as model:
            
            # fix analyzed reactions to grid point value:
            self.diatom.fix_growth_rates(model, grid_point)

            # analyze feasible point
            if not self.fva_reactions:
                raise RuntimeError('No reactions selected for fva and clustering!')
                    
            #print(f"running FVA on grid point: {grid_point}")
                
            rxn_fva = flux_variability_analysis(model, reaction_list=self.fva_reactions) # type: ignore              
            rxn_fva = rxn_fva.loc[self.fva_reactions, :] # just to make sure reactions are in the same order as fva_reactions
            minimum_values = rxn_fva["minimum"].to_numpy()
            maximum_values = rxn_fva["maximum"].to_numpy()

            #print("translating to qualitative vector..")
            qualitative_vector = qual_translate(minimum_values, maximum_values, delta=delta)
            fva_results = rxn_fva.values

            fva_tuple = (list(qualitative_vector), fva_results)
            self.diatom.io.save_fva_result(grid_point, fva_tuple)

            return fva_tuple


    # ================================================== QUANTITATIVE GRID ANALYSIS ==================================================


    PointList = list[float | int] | list[int] | list[float]
    def quan_FCA(self, grid_x: PointList, grid_y: PointList, reaction_ids: tuple[str, str]) -> None:
        """Performs quantitative Flux Coupling Analysis on two reactions (rxns_analysis) on points of a sub-grid defined by points grid_x, grid_y.
        
        Stores an attribute self.qFCA containing a dataframe with the following columns: 
            flux_rxns_analysis[0], flux_rxn_analysis[1], FVA (str: minimum or maximum), point (coordinates of point)
        """
        assert len(reaction_ids) == 2

        feasible_points = self.diatom.grid.points[self.diatom.grid.feasible_points]
        reaction_id_0 = reaction_ids[0]
        reaction_id_1 = reaction_ids[1]

        print('Quantitative Flux Coupling analysis \n Initializing grid...')

        analyze_points = []
        # Match points defined by the user in grid_x, grid_y to specific points on the grid
        for y in grid_y:
            for x in grid_x:
                search_point = np.array([x, y])
                distances = np.linalg.norm(feasible_points-search_point, axis=1)
                min_index = np.argmin(distances)
                analyze_points.append(min_index)
                #print(f"The closest point to {search_point} is {feasible_points[min_index]}, at a distance of {distances[min_index]}")

        qFCA_data = []

        for point in analyze_points:
            grid_point = feasible_points[point]
            
            #print(f'Selected point: {grid_point}')

            with self.diatom.model as model:
                # update bounds nad objectives
                self.diatom.fix_growth_rates(model, grid_point)

                # define limit reactions based on theoretical max-min defined from model
                fva_result = flux_variability_analysis(model, reaction_list = [reaction_id_0])
                min_value = float(fva_result['minimum'].iloc[0])
                max_value = float(fva_result['maximum'].iloc[0])
                values_rxn_ref = np.linspace(min_value, max_value, num=50)

                reaction_0 = cast(Reaction, model.reactions.get_by_id(reaction_id_0))
                
                for value in values_rxn_ref:
                    reaction_0.bounds = (value, value)
                    fva_result = flux_variability_analysis(model, reaction_list = [reaction_id_1])
                    
                    for bound in fva_result: # [minimum, maximum]
                        qFCA_data.append({
                            reaction_id_0: value,
                            reaction_id_1: fva_result[bound].iloc[0],
                            'FVA': bound,
                            'point': f"{grid_point[0]:.3f}, {grid_point[1]:.3f}"
                        })

        self.qFCA = pd.DataFrame(qFCA_data)

