from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from tqdm import tqdm
from cobra import Reaction
from cobra.flux_analysis import flux_variability_analysis


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
"""Mapping between numeric qualitative codes and symbolic labels.

The numeric codes are produced by `qual_translate` and represent qualitative
flux variability states derived from FVA minimum/maximum values.
"""


def qual_translate(fmin: np.ndarray, fmax: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Translate FVA min/max values into qualitative flux states.

    Compares minimum and maximum flux values obtained from FVA and assigns
    each reaction to a qualitative category based on sign, variability, and
    numerical tolerance.

    Parameters
    ----------
    fmin : np.ndarray
        Array of minimum flux values from FVA.

    fmax : np.ndarray
        Array of maximum flux values from FVA.

    delta : float, default=1e-4
        Numerical tolerance used to determine equality to zero.

    Returns
    -------
    np.ndarray
        Array of numeric qualitative codes. These codes can be translated
        into symbolic labels using `CATEGORY_DICT`.
    """

    same_value = np.abs(fmax - fmin) < eps
    pos_max = fmax > eps
    neg_max = fmax < -eps
    pos_min = fmin > eps
    neg_min = fmin < -eps
    zero_max = np.abs(fmax) <= eps
    zero_min = np.abs(fmin) <= eps

    # order of evaluation here is VERY IMPORTANT
    conditions = [
        neg_min & neg_max & same_value,
        neg_min & neg_max,
        neg_min & zero_max,
        zero_min & zero_max, 
        zero_min & pos_max,
        pos_min & pos_max & same_value, 
        pos_min & pos_max, 
        neg_min & pos_max, 
    ]

    choices = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

    return np.select(conditions, choices, default=5.0)


class DiatomAnalyze():
    """Analysis class for polytope construction and flux-based analyses.

    This class encapsulates all analysis steps that require interaction with
    the metabolic model, including:
    - Construction of low-dimensional projections of the feasible flux space.
    - Qualitative Flux Variability Analysis (qFVA) over grid-sampled points.
    - Quantitative Flux Coupling Analysis (qFCA).

    Parameters
    ----------
    diatom : Diatom
        Parent diatom object providing access to the metabolic model,
        grid sampler, and I/O utilities.

    Attributes
    ----------
    polytope : BaseGeometry
        2D geometric representation of the projected feasible flux space.

    n_angles : int, default=360
        Number of angular directions used to sample the polytope boundary.
        Higher values improve boundary resolution at increased computational cost.

    analyzed_reactions : tuple[str, str]
        Pair of reaction IDs used to construct the 2D polytope projection.

    fva_reactions : list[str]
        Reaction IDs selected for Flux Variability Analysis.

    fva_results : np.ndarray, shape (n_points, n_reactions, 2)
        FVA results over analyzed grid points.
  
    qual_vector_df : pd.DataFrame
        DataFrame containing qualitative flux categories for each reaction and grid point.

    category_dict : dict[float, str]
        Mapping between numeric qualitative codes and symbolic labels.

    qFCA : pd.DataFrame
        Results of quantitative Flux Coupling Analysis.
    """
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom

        self.analyzed_reactions: tuple[str, str] 
        self.fva_reactions: list[str] = []      
        self.fva_results: np.ndarray # shape: (n_points, n_reactions, 2)

        self.qual_vector: pd.DataFrame   
        self.category_dict: dict[float, str] = CATEGORY_DICT
        self._empty_qual_vector: list[float] | None = None
        self._empty_fva_result: np.ndarray | None = None

        self.use_pfba: bool
        self.pfba_fraction: float

        self.qFCA: pd.DataFrame = pd.DataFrame()


    def qualitative_analysis(
        self,  
        x_limits: tuple[float, float] = (-np.inf, np.inf),  
        y_limits: tuple[float, float] = (-np.inf, np.inf), 
        only_load: bool = False,  
        eps: float = 1e-9,
    ) -> None:
        """Run qualitative FVA over selected grid points.

        Computes qualitative reaction categories and flux variability analysis (FVA)
        results for grid points within the specified coordinate bounds.

        If feasible grid points have been previously computed, the function will load those
        points in order to not recompute FVA again.

        Parameters
        ----------
        x_limits : tuple[float, float], optional
            Inclusive lower and upper bounds on the x coordinate used to filter grid points.
        y_limits : tuple[float, float], optional
            Inclusive lower and upper bounds on the y coordinate used to filter grid points.
        only_load : bool, default=False
            If True, restricts the analysis to loaded reactions only.

        Side Effects
        ------------
        - Updates `self.qual_vector_df` with qualitative reaction categories for each analyzed grid point.
        - Updates `self.fva_results` with FVA min/max values per reaction.
        - Updates `self.diatom.grid.analyzed_points` to set the subset of grid points used in the analysis.
        """
        self.diatom._require(grid_points=True)
        print("Running qualitative fva over grid feasible points...")

        points = self.diatom.grid.points               
        feasible_points = self.diatom.grid.feasible_points

        # select points for analysis
        filtered = (points[:, 0] > x_limits[0]) & (points[:, 0] < x_limits[1]) & (points[:, 1] > y_limits[0]) & (points[:, 1] < y_limits[1])
        analyzed_points = filtered & feasible_points
        self.diatom.grid.analyzed_points = analyzed_points

        # perform analysis
        points = points[analyzed_points, :]    
        df_index = np.where(analyzed_points)[0]

        fva_tuples = self._calculate_qual_vectors(points, only_load = only_load, eps=eps)
        qual_vector_list, fva_results = map(list, zip(*fva_tuples))    

        self.qual_vector = pd.DataFrame(np.array(qual_vector_list), columns=self.fva_reactions, index=df_index)
        #self.diatom.io.save_qual_df()    
        fva_results = np.dstack(fva_results)
        fva_results = np.rollaxis(fva_results, -1)
        self.fva_results = fva_results

        print("Done!\n")         


    def _calculate_qual_vectors(self, grid_points: np.ndarray, only_load: bool, eps: float) -> list[tuple]:
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

        print("Analyzing point feasibility....")
        n_points = grid_points.shape[0]
        if only_load:
            fva_tuples = []
            for grid_point in tqdm(grid_points, total=n_points):
                loaded = self._load_if_stored(grid_point, eps=eps)
                if loaded is not None:
                    fva_tuples.append(loaded)
        else:
            fva_tuples = [self._analyze_point(grid_point, eps) for grid_point in tqdm(grid_points, total = n_points)] 

        return fva_tuples
    
    
    def _load_if_stored(self, grid_point: np.ndarray, eps: float):
        """Load previously computed FVA results for a grid point.

        Attempts to retrieve stored FVA results for the given grid point. 
        If no stored result is found, a placeholder result filled with NaNs is returned.
        """
        loaded = self.diatom.io.load_point(grid_point, "qual_fva")

        if isinstance(loaded, np.ndarray):
            qualitative_vector = self._compute_qual_from_fva(loaded, eps)
            return (qualitative_vector, loaded)

        # placeholder
        if self._empty_qual_vector is None or self._empty_fva_result is None:
            n_rxns = len(self.fva_reactions)
            self._empty_qual_vector = [np.nan] * n_rxns
            self._empty_fva_result = np.full((n_rxns, 2), np.nan)

        return (self._empty_qual_vector, self._empty_fva_result)


    def _compute_qual_from_fva(self, fva_results: np.ndarray, eps: float) -> list:
        """Translate stored FVA min/max values into a qualitative vector."""
        minimum_values = fva_results[:, 0]
        maximum_values = fva_results[:, 1]
        qualitative_vector = qual_translate(minimum_values, maximum_values, eps=eps)
        return list(qualitative_vector)
    

    def _analyze_point(self, grid_point: np.ndarray, eps: float) -> tuple:
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

        eps : float, default=1e-9
            Numerical tolerance used when translating flux variability ranges into qualitative categories. 

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
        if isinstance(loaded_point, np.ndarray):
            qualitative_vector = self._compute_qual_from_fva(loaded_point, eps=eps)
            return (qualitative_vector, loaded_point)
        
        if not self.fva_reactions:
            raise RuntimeError('No reactions selected for fva and clustering!')
        
        with self.diatom.model as model:
            # fix analyzed reactions to grid point value:
            self.diatom.fix_growth_rates(model, grid_point)

            if self.use_pfba:
                self.diatom.apply_pfba_constraint(model, fraction_of_optimum=self.pfba_fraction)
                
            # analyze feasible point
            rxn_fva = flux_variability_analysis(model, reaction_list=self.fva_reactions) # type: ignore              
            rxn_fva = rxn_fva.loc[self.fva_reactions, :] # just to make sure reactions are in the same order as fva_reactions
            fva_results = rxn_fva.values
            self.diatom.io.save_fva_result(grid_point, fva_results)

        qualitative_vector = self._compute_qual_from_fva(fva_results, eps=eps)

        return (qualitative_vector, fva_results)


    # ================================================== QUANTITATIVE GRID ANALYSIS ==================================================


    PointList = list[float | int] | list[int] | list[float]
    def quan_FCA(self, grid_x: PointList, grid_y: PointList, reaction_ids: tuple[str, str]) -> None:
        """Perform quantitative Flux Coupling Analysis (qFCA) on a subgrid.

        Evaluates the coupling between two reactions by fixing the flux of the
        first reaction across its feasible range and computing the resulting
        FVA bounds of the second reaction at selected grid points.

        Parameters
        ----------
        grid_x : list[float | int]
            X-coordinates of the subgrid points to analyze.

        grid_y : list[float | int]
            Y-coordinates of the subgrid points to analyze.

        reaction_ids : tuple[str, str]
            Pair of reaction IDs `(reference_reaction, coupled_reaction)`.

        Attributes Set
        --------------
        qFCA : pd.DataFrame
            DataFrame containing quantitative coupling results with columns:
            - flux of reference reaction
            - flux of coupled reaction
            - FVA bound type (minimum or maximum)
            - grid point coordinates
        """
        assert len(reaction_ids) == 2
        self.diatom._require(grid_points=True)

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

