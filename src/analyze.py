import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from tqdm import tqdm
from cobra import Reaction
from cobra.flux_analysis import flux_variability_analysis

from .constants import NON_ZERO_TOLERANCE, CATEGORY_DICT

if TYPE_CHECKING:
    from .metabolic_experiment import MetabolicExperiment


class Analyze():
    """Analysis class for flux-based analyses.

    This class encapsulates all analysis steps that require interaction with
    the metabolic model, including:
    - Qualitative Flux Variability Analysis (qFVA) over grid-sampled points.
    - Quantitative Flux Coupling Analysis (qFCA).

    Parameters
    ----------
    parent_class : MetabolicExperiment
        Parent class object providing access to the metabolic model,
        grid sampler, and I/O utilities.

    Attributes
    ----------
    analyzed_reactions : tuple[str, str]
        Pair of reaction IDs used to construct the 2D polytope projection.

    fva_reactions : list[str]
        Reaction IDs selected for Flux Variability Analysis.

    fva_results : np.ndarray, shape (n_points, n_reactions, 2)
        FVA results over analyzed grid points.
  
    qualitative_matrix : pd.DataFrame
        DataFrame containing qualitative flux categories for each reaction (columns) 
        and grid point (indexes).

    category_dict : dict[float, str]
        Mapping between numeric qualitative codes and symbolic labels.

    use_pfba: bool
        If True, the model is restricted to parsimonious flux distributions.

    pfba_fraction: float
        Fraction of optimum to be used if parsimonious solutions are required.

    qFCA : pd.DataFrame
        Results of quantitative Flux Coupling Analysis.
    """
    def __init__(self, parent_class: "MetabolicExperiment"):
        self.parent_class = parent_class

        self.analyzed_reactions: tuple[str, str] 
        self.fva_reactions: list[str] = []      
        self.fva_results: np.ndarray # shape: (n_points, n_reactions, 2)

        self.qualitative_matrix: pd.DataFrame   
        self.category_dict: dict[float, str] = CATEGORY_DICT
        self._empty_qualitative_matrix: list[float] | None = None
        self._empty_fva_result: np.ndarray | None = None

        self.use_pfba: bool
        self.pfba_fraction: float

        self.qFCA: pd.DataFrame


    def qualitative_analysis(
        self,  
        x_limits: tuple[float, float] = (-np.inf, np.inf),  
        y_limits: tuple[float, float] = (-np.inf, np.inf), 
        only_load: bool = False,  
        non_zero_tolerance: float = NON_ZERO_TOLERANCE,
    ) -> None:
        """Run qualitative FVA over selected grid points.

        Computes flux variability analysis (FVA) results for grid points within the specified 
        coordinate bounds, and then calculates qualtitative states for each reaction.

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

        non_zero_tolerance: float, default=1e-6
            Flux values under this parameter are considered zero.

        Attributes Set
        --------------
        - Updates `self.qualitative_matrix` with qualitative reaction categories for each analyzed grid point.
        - Updates `self.fva_results` with FVA min/max values per reaction.
        - Updates `self.parent_class.grid.analyzed_points` to set the subset of grid points used in the analysis.
        """
        self.parent_class._require(grid_points=True)
        logging.info("Running qualitative fva over grid feasible points...")

        points = self.parent_class.grid.points               
        feasible_points = self.parent_class.grid.feasible_points

        # select points for analysis
        filtered = (
            (points[:, 0] > x_limits[0]) & 
            (points[:, 0] < x_limits[1]) & 
            (points[:, 1] > y_limits[0]) & 
            (points[:, 1] < y_limits[1])
        )
        analyzed_points = filtered & feasible_points
        self.parent_class.grid.analyzed_points = analyzed_points

        points = points[analyzed_points, :]    
        df_index = np.where(analyzed_points)[0]

        # check for reactions selected for FVA and clustering
        if not self.fva_reactions:
            logging.warning(
                "No reactions previously selected for FVA and clustering!\n"
                "Setting reactions for analysis...\n"
            )
            self.parent_class._set_non_blocked_reactions()  
            fva_reactions = list(self.parent_class.non_blocked)  
            fva_reactions.sort()
            self.fva_reactions = fva_reactions

        logging.info(f"Number of reactions set for analysis: {len(self.fva_reactions)}")

        qualitative_vectors, fva_results = zip(*self._calculate_qual_vectors(
            points, only_load=only_load, non_zero_tolerance=non_zero_tolerance,
        )) 

        self.qualitative_matrix = pd.DataFrame(
            qualitative_vectors, 
            columns=self.fva_reactions, 
            index=df_index
        )
 
        self.fva_results = np.rollaxis(np.dstack(fva_results), -1)

        logging.info("Done!\n")         


    def _calculate_qual_vectors(
        self, grid_points: np.ndarray, only_load: bool, non_zero_tolerance: float,
    ) -> list[tuple]:
        """Calculate qualitative FVA vectors for a set of grid points.

        Iterates over grid points and calculates qualitative FVA vectors. Each element in the returned 
        list is a tuple `(qualitative_vector, fva_result)` for a point.
        """
        logging.info("Analyzing point feasibility....")
        n_points = grid_points.shape[0]
        if only_load:
            fva_tuples = []
            for grid_point in tqdm(grid_points, total=n_points):
                loaded = self._load_if_stored(grid_point, non_zero_tolerance=non_zero_tolerance)
                if loaded is not None:
                    fva_tuples.append(loaded)
        else:
            fva_tuples = [
                self._analyze_point(grid_point, non_zero_tolerance) 
                for grid_point in tqdm(grid_points, total = n_points)
            ] 

        return fva_tuples
    
    
    def _load_if_stored(self, grid_point: np.ndarray, non_zero_tolerance: float):
        """Load previously computed FVA results for a grid point.

        Attempts to retrieve stored FVA results for the given grid point. 
        If no stored result is found, a placeholder result filled with NaNs is returned.
        """
        loaded = self.parent_class.io.load_point(grid_point, "qual_fva")

        if isinstance(loaded, np.ndarray):
            qualitative_vector = self._qualitative_translate(loaded, non_zero_tolerance)
            return (qualitative_vector, loaded)

        # placeholder
        if self._empty_qualitative_matrix is None or self._empty_fva_result is None:
            n_rxns = len(self.fva_reactions)
            self._empty_qualitative_matrix = [np.nan] * n_rxns
            self._empty_fva_result = np.full((n_rxns, 2), np.nan)

        return (self._empty_qualitative_matrix, self._empty_fva_result)


    @staticmethod
    def _qualitative_translate(fva_results: np.ndarray, non_zero_tolerance: float) -> np.ndarray:
        """Translate FVA min/max values into qualitative flux states.

        Compares minimum and maximum flux values obtained from FVA and assigns
        to each reaction a qualitative category based on sign, variability, and
        numerical tolerance.

        Parameters
        ----------
        fva_results : np.ndarray
            Array containing minimum and maximum flux values from FVA.

        non_zero_tolerance : float, default=1e-6
            Numerical tolerance used to determine equality to zero. Fluxes whose absolute
            value are under this parameter are considered zero.

        Returns
        -------
        np.ndarray
            Array of numeric qualitative codes. These codes can be translated
            into symbolic labels using `CATEGORY_DICT`.
        """
        fmin = fva_results[:, 0]
        fmax = fva_results[:, 1]

        same_value = np.abs(fmax - fmin) < non_zero_tolerance
        pos_max = fmax > non_zero_tolerance
        neg_max = fmax < -non_zero_tolerance
        pos_min = fmin > non_zero_tolerance
        neg_min = fmin < -non_zero_tolerance
        zero_max = np.abs(fmax) <= non_zero_tolerance
        zero_min = np.abs(fmin) <= non_zero_tolerance

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
    

    def _analyze_point(self, grid_point: np.ndarray, non_zero_tolerance: float) -> tuple:
        """Analyze a single grid point via Flux Variability Analysis (FVA).

        For a given grid point in the projected flux space, this method fixes the
        analyzed reactions to the coordinates defined by `grid_point`, optionally
        applies parsimonious constraints, and performs FVA over the selected reactions.

        If FVA results for the grid point have been previously computed and stored,
        they are loaded from disk and reused to avoid recomputation.

        Parameters
        ----------
        grid_point : np.ndarray
            Coordinates of the grid point in the analyzed 2D projection. These
            values are used to constrain the corresponding reactions in the model.

        non_zero_tolerance : float
            Numerical tolerance used to determine whether flux values are treated
            as zero when translating quantitative FVA results into qualitative
            categories.

        Returns
        -------
        tuple
            A tuple `(qualitative_vector, fva_results)` where:
            - `qualitative_vector` is a 1D array of numeric qualitative codes
              obtained by translating FVA min/max values.
            - `fva_results` is a 2D array of shape `(n_reactions, 2)` containing
              minimum and maximum flux values for each analyzed reaction.
        """
        loaded_point = self.parent_class.io.load_point(grid_point, "qual_fva")
        if isinstance(loaded_point, np.ndarray):
            qualitative_vector = self._qualitative_translate(
                loaded_point, non_zero_tolerance=non_zero_tolerance,
            )
            return (qualitative_vector, loaded_point)
        
        if not self.fva_reactions:
            raise RuntimeError('No reactions selected for fva and clustering!')
        
        with self.parent_class.model as model:
            self.parent_class.fix_flux_rates(model, grid_point)

            if self.use_pfba:
                self.parent_class.apply_pfba_constraint(
                    model, fraction_of_optimum=self.pfba_fraction,
                )
                
            # analyze feasible point
            rxn_fva = flux_variability_analysis(model, reaction_list=self.fva_reactions) # type: ignore              
            rxn_fva = rxn_fva.loc[self.fva_reactions, :] # makes sure reactions are in the same order as fva_reactions
            fva_results = rxn_fva.values
            self.parent_class.io.save_fva_result(grid_point, fva_results)

        qualitative_vector = self._qualitative_translate(
            fva_results, non_zero_tolerance=non_zero_tolerance,
        )

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
        self.parent_class._require(grid_points=True)

        feasible_points = self.parent_class.grid.points[self.parent_class.grid.feasible_points]
        reaction_id_0 = reaction_ids[0]
        reaction_id_1 = reaction_ids[1]

        logging.info('Quantitative Flux Coupling analysis \n Initializing grid...')

        analyze_points = []
        # Match points defined by the user in grid_x, grid_y to specific points on the grid
        for y in grid_y:
            for x in grid_x:
                search_point = np.array([x, y])
                distances = np.linalg.norm(feasible_points-search_point, axis=1)
                min_index = np.argmin(distances)
                analyze_points.append(min_index)
                logging.debug(
                    f"The closest point to {search_point} is {feasible_points[min_index]}, "
                    f"at a distance of {distances[min_index]}"
                )

        qFCA_data = []

        for point in analyze_points:
            grid_point = feasible_points[point]
            with self.parent_class.model as model:
                # update bounds nad objectives
                self.parent_class.fix_flux_rates(model, grid_point)

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

