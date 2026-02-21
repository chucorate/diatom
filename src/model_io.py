import logging
import pickle
import hashlib
import json
import platform
from typing import TYPE_CHECKING, Literal, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cobra
import cobra.io
from cobra import Model
import xlsxwriter

if TYPE_CHECKING:
    from diatom.diatom import Diatom


MODEL_DIRECTORY = Path("models")
RESULTS_DIRECTORY = Path("result_files")


def file_hash(path):
    with open(MODEL_DIRECTORY / path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
    

def load_model(model_name: str, solver: str = 'gurobi', **kwargs) -> Model:
    """Loads a COBRA model from an SBML file using the specified solver.

    Parameters
    ----------
    model_name : str
        SBML filename (relative to `model_directory`).
    model_directory : str, default=MODEL_DIR
        Directory containing SBML models.
    solver : str, default='gurobi'
        Solver backend to attach to the model.
    **kwargs
        Additional arguments passed to `cobra.io.read_sbml_model`.

    Returns
    -------
    cobra.Model
        Loaded COBRA model with configured solver.
    """
    path = MODEL_DIRECTORY / model_name
    model = cobra.io.read_sbml_model(path, solver=solver, **kwargs)

    return model


def save_models(model_dict: dict[str, Model]) -> None:
    """Saves all COBRA models in `model_dict` to `model_directory`."""    
    for model_name, model in model_dict.items():
        filename = MODEL_DIRECTORY / f"{model_name}.xml"
        cobra.io.write_sbml_model(model, filename)
        logging.info(f'model {model_name} stored')


def canonicalize(obj: Any) -> Any:
    """Standarizes the metadata dictionary in order to ensure json format
    compatibility, and to avoid producing different hashes with the same data
    inputed in different ways."""
    if isinstance(obj, dict):
        return {x: canonicalize(obj[x]) for x in sorted(obj)}

    if isinstance(obj, (list, tuple, set)):
        obj = [canonicalize(x) for x in obj]
        return list(sorted(obj, key=repr))

    if isinstance(obj, float | int):
        obj = float(round(obj, 6)) if abs(obj) > 1e-6 else 0.0
        return str(obj)

    if isinstance(obj, (str, bool)) or obj is None:
        return obj

    raise TypeError(f"Unhashable type in metadata: {obj: type(obj)}")


class ModelIO():
    """Handle filesystem I/O for analysis results associated with a model instance.

    This class is responsible for:
    - constructing directory.
    - saving and loading per-grid-point results and clustering data.
    - exporting dataframes and plots.
    """
    def __init__(self, modelclass: "Diatom", model_name: str):
        self.modelclass = modelclass
        self.model_name: str = model_name
        self.save_files: bool = False
        self.load_files: bool = False
        self.experiment_tag: str


    @property
    def sampling_hash(self) -> str:
        """Hash used to uniquely identify a sampling instance by its specified parameters."""
        canonical = json.dumps(self.modelclass.metadata)
        hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        initial_label = f"{self.experiment_tag}_#"
        hash = initial_label + hash

        return hash
    

    @property
    def numeric_identifier(self) -> str:
        """String used to specify numeric, non_biologic parameters for data identification.
        
        Assumes a sampling hash has already been set, and using it alongside the numeric identifier
        files can be saved and loaded safely without overwriting previous results."""
        n_sampling_angles = self.modelclass.projection.n_sampling_angles
        n_partitions = self.modelclass.grid.n_partitions
        n_clusters = self.modelclass.clustering.grid_n_clusters

        identifier = f"SA{n_sampling_angles}_NP{n_partitions}_NC{n_clusters}"
        return identifier


    @property
    def analyzed_tuple_string(self) -> str:
        """String identifier for the current analyzed reaction tuple.
        
        Raises an assertion error if analyzed reactions have not been set."""
        reaction1, reaction2 = self.modelclass.analyze.analyzed_reactions
        assert not (reaction1 == "" or reaction2 == "")
        return f"{reaction1}_{reaction2}"


    @property
    def results_directory(self) -> Path:
        """General results directory."""
        path = RESULTS_DIRECTORY / self.model_name / self.analyzed_tuple_string / self.sampling_hash
        return path 
    

    @property
    def _analysis_directory(self) -> Path:
        """"Directory that holds analysis data.
        
        Only gets created when called from a method that saves or loads data."""
        path = self.results_directory / "analysis"
        path.mkdir(parents=True, exist_ok=True)
        return path
    

    @property
    def _dataframe_directory(self) -> Path:
        """"Directory that holds clustering dataframe data.
        
        Only gets created when called from a method that saves or loads data."""
        path = self.results_directory / "dataframes"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    
    @property
    def _plots_directory(self) -> Path:
        """"Directory that holds plots.
        
        Only gets created when called from a method that saves or loads data."""
        path = self.results_directory / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path
    

    def write_metadata(self) -> None:
        """Create or validate the metadata.json file for the current experiment.

        If the file does not exist, it is created.
        If it exists, its biological identity is validated against the
        current experiment configuration. A mismatch raises an error.
        """
        if not (self.save_files or self.load_files):
            return

        path = self.results_directory
        path.mkdir(parents=True, exist_ok=True)

        metadata_path = path / "metadata.json"
        metadata = {
            "experiment_hash": self.sampling_hash,
            "model_file": self.modelclass.model_id,
            "model_hash": self.modelclass.metadata["model_hash"],
            "reaction_tuple": self.modelclass.metadata["reaction_tuple"],
            "metabolites": self.modelclass.metadata["metabolites"],
            "reactions": self.modelclass.metadata["reactions"],
            "bound_constraints": self.modelclass.metadata["bound_constraints"],
            "flux_constraints": self.modelclass.metadata["flux_constraints"],
            "use_pfba": self.modelclass.analyze.use_pfba,
            "pfba_fraction_of_optimum": self.modelclass.analyze.pfba_fraction,
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "created_at": datetime.now().isoformat(),
            }
        }

        # validates already existing metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                existing = json.load(f)

            keys_to_compare = [
                "model_hash",
                "reaction_tuple",
                "metabolites",
                "reactions",
                "bound_constraints",
                "flux_constraints",
                "use_pfba",
                "pfba_Fraction_of_optimum",
            ]

            for key in keys_to_compare:
                existing_key = existing.get(key)
                metadata_key = metadata.get(key)
                if existing_key != metadata_key:
                    raise RuntimeError(
                        f"Metadata mismatch detected in {metadata_path}.\n"
                        f"Field '{key}' differs from stored experiment:\n"
                        f"Existing key: {existing_key}, current_key: {metadata_key}"
                        "\n\nThere exists a hash collision."
                    )
            return

        # writes new metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)



    # =================================================== ANALYSIS DIRECTORY ===================================================

    
    @staticmethod
    def _format_coord(x: float) -> str:
        return f"{round(x, 6):.6f}".replace('.', 'p').replace('-', 'm')
    

    def _coordinates_to_filename(self, grid_point: np.ndarray) -> str:
        """Formats a floating point coordinate into a filesystem-safe string.""" 
        x_str = self._format_coord(grid_point[0])
        y_str = self._format_coord(grid_point[1])
        return f"x_{x_str}_y_{y_str}.pkl"
    

    def _get_directory(self, subdirectory: Literal["feasibility", "qual_fva", "clustering"]) -> Path:
        """Returns the analysis subdirectory requested for the current grid. 
        If it doesn't exists, the method creates it."""
        Lx, Ly = self.modelclass.grid.grid_dimensions
        grid_dim = f"Lx_{Lx:.4f}_Ly_{Ly:.4f}"

        directory = self._analysis_directory / grid_dim / subdirectory
        directory.mkdir(parents=True, exist_ok=True) 

        return directory


    def _get_point_directory(self, grid_point: np.ndarray, subdirectory: Literal["feasibility", "qual_fva"]) -> Path:
        directory = self._get_directory(subdirectory)
        filename = f"{subdirectory}_{self._coordinates_to_filename(grid_point)}"
        return directory / filename
    

    def load_point(self, grid_point: np.ndarray, analysis: Literal["feasibility", "qual_fva"]) -> bool | tuple | None:
        """Load per-point analysis results from disk.

        Returns
        -------
        bool | tuple | None
            - feasibility: bool
            - qualitative FVA: tuple
            - None if the point is not saved.
        """
        if not self.load_files:
            return

        path = self._get_point_directory(grid_point, analysis)
        if not path.exists():
            logging.debug(f"Directory doesn't exists: {path}")
            return

        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
            return loaded_data
        

    def save_feasible_point(self, grid_point: np.ndarray, is_feasible: bool) -> None:
        if not self.save_files:
            return
        
        # making the directory to store the point
        path = self._get_point_directory(grid_point, "feasibility")
        
        with open(path, "wb") as f:
            pickle.dump(is_feasible, f)


    def save_fva_result(self, grid_point: np.ndarray, fva_results: np.ndarray) -> None:
        if not self.save_files:
            return

        # making the directory to store the point
        path = self._get_point_directory(grid_point, "qual_fva")
        
        with open(path, "wb") as f:
            pickle.dump(fva_results, f)


    def get_clusters_directory(self) -> Path:
        filename = f"Clusters_{self.numeric_identifier}.pkl"
        directory = self._get_directory("clustering")
        return directory / filename


    def load_clusters(self) -> tuple | None:
        if not self.load_files:
            return
        
        path = self.get_clusters_directory()
        if not path.exists():
            return

        with open(path, 'rb') as f:
            clusters_tuple = pickle.load(f)
            return clusters_tuple

    
    def save_clusters(self, n_clusters: int, clusters: np.ndarray) -> None:
        if not self.save_files:
            return
        
        clusters_tuple = (n_clusters, clusters)
        # making the directory to store clusters
        path = self.get_clusters_directory()
        with open(path, "wb") as f:
            pickle.dump(clusters_tuple, f)


    # =================================================== DATAFRAME DIRECTORY ===================================================


    def save_cluster_df(
        self, 
        df: pd.DataFrame, 
        type: Literal["Qualitative_profiles", "Metrics_per_reaction", "Global_metrics"], 
        index: bool = False,
        reaction_len: int = -1,
        metric_list: list[str] | None = None,
        overwrite: bool = False,
    ) -> None: 
        if not self.save_files:
            return
        
        file = f"{type}_{self.numeric_identifier}_NR{reaction_len}"
        if metric_list is not None:
            file += f"_M{len(metric_list)}"

        path = self._dataframe_directory / f"{file}.csv"
        if path.exists() and not overwrite:
            logging.warning(f"Skipping existing file: {path.name}")
            return
        
        df.to_csv(path, index=index, encoding='utf-8')


    def merge_to_excel(self, df_dict: dict[str, pd.DataFrame]) -> None:
        if not self.save_files:
            return
        
        path = RESULTS_DIRECTORY / self.model_name / "supplementary" / f"{self.analyzed_tuple_string}.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for df_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=df_name, index=False)


    # =================================================== PLOT DIRECTORY ===================================================


    def save_plot_path(self, extra_label: str | None = None) -> Path | None:
        if not self.save_files:
            return
        
        plot_label = f"_{extra_label}" if extra_label is not None else ""
        path = self._plots_directory / f"Plot{plot_label}_{self.numeric_identifier}.png"
        return path

