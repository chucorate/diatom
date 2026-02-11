from typing import TYPE_CHECKING, Literal
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cobra import Model
import cobra.io

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem
    from diatom.diatom import Diatom


MODEL_DIR = "models"

SAVE_POINTS_DIR = "models/points"


def load_model(model_name: str, model_directory: str = MODEL_DIR, solver: str = 'gurobi', **kwargs) -> Model:
    '''Loads a COBRA model from an SBML file using the specified solver.'''
    path = Path(model_directory) / model_name
    model = cobra.io.read_sbml_model(path, solver=solver, **kwargs)
    model.solver.configuration.threads = 0

    return model 


def save_models(model_dict: dict[str, Model], model_directory: str = MODEL_DIR) -> None:
    '''Saves all COBRA models in "model_dict" to "model_directory".'''    
    output_dir = Path(model_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in model_dict.items():
        filename = output_dir / f"{model_name}.xml"
        cobra.io.write_sbml_model(model, filename)
        print(f'model {model_name} stored')


class ModelIO():
    def __init__(self, modelclass: "BaseEcosystem | Diatom", model_name: str):
        self.modelclass = modelclass
        self.directory: Path | None = None
        self.model_name: str = model_name

    
    @property
    def grid_dimensions(self) -> np.ndarray:
        return self.modelclass.grid.grid_dimensions
    

    @property
    def points_per_axis(self) -> tuple[int, int]:
        return self.modelclass.grid.points_per_axis
    

    @property
    def analyzed_tuple_string(self) -> str:
        reaction1, reaction2 = self.modelclass.analyze.analyzed_reactions
        assert not (reaction1 == "" or reaction2 == "")
        return f"{reaction1}_{reaction2}"


    @property
    def results_directory(self) -> Path:
        path = Path("result_files")
        return path / self.model_name / self.analyzed_tuple_string
    

    @property
    def analysis_directory(self) -> Path:
        path = self.results_directory / "analysis"
        return path
    

    @property
    def dataframe_directory(self) -> Path:
        path = self.results_directory / "dataframes"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    
    @property
    def plots_directory(self) -> Path:
        path = self.results_directory / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path


    # =================================================== ANALYSIS DIRECTORY ===================================================

    
    @staticmethod
    def _format_coord(x: float) -> str:
        # Convierte el float a string válido para nombre de archivo, con precisión fija
        return f"{round(x, 6):.6f}".replace('.', 'p').replace('-', 'm')
    

    def _coordinates_to_filename(self, grid_point: np.ndarray) -> str:
        x_str = self._format_coord(grid_point[0])
        y_str = self._format_coord(grid_point[1])
        return f"x_{x_str}_y_{y_str}.pkl"
    

    def get_directory(self, subdirectory: Literal["feasibility", "qual_fva", "clustering"]) -> Path:
        Lx, Ly = self.grid_dimensions
        grid_dim = f"Lx_{Lx:.4f}_Ly_{Ly:.4f}"

        directory = self.analysis_directory / grid_dim / subdirectory
        
        if self.directory is None:
            self.directory = self.analysis_directory / grid_dim 

        directory.mkdir(parents=True, exist_ok=True) 

        return directory


    def get_point_directory(self, grid_point: np.ndarray, subdirectory: Literal["feasibility", "qual_fva"]) -> Path:
        directory = self.get_directory(subdirectory)
        filename = f"{subdirectory}_{self._coordinates_to_filename(grid_point)}"

        return directory / filename
    

    def is_point_saved(self, grid_point: np.ndarray, subdirectory: Literal["feasibility", "qual_fva"]) -> bool:
        return self.get_point_directory(grid_point, subdirectory).exists()
    

    def load_point(self, grid_point: np.ndarray, analysis: Literal["feasibility", "qual_fva"]) -> bool | tuple | None:
        if not self.is_point_saved(grid_point, analysis):
            #print(f"directory doesn't exists")
            return None 
        
        path = self.get_point_directory(grid_point, analysis)

        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
            if analysis == "feasibility":
                return loaded_data["is_feasible"]
            return loaded_data["fva_tuple"]
        

    def save_feasible_point(self, grid_point: np.ndarray, is_feasible: bool, update_bounds: bool = True) -> None:
        point_dict = {
            "is_feasible": is_feasible,
            "update_bounds": update_bounds,
        }
        
        # making the directory to store the point
        path = self.get_point_directory(grid_point, "feasibility")
        
        with open(path, "wb") as f:
            pickle.dump(point_dict, f)


    def save_fva_result(self, grid_point: np.ndarray, fva_tuple: tuple, update_bounds: bool = True) -> None:
        point_dict = {
            "fva_tuple": fva_tuple,
            "update_bounds": update_bounds
        }
        
        # making the directory to store the point
        path = self.get_point_directory(grid_point, "qual_fva")
        
        with open(path, "wb") as f:
            pickle.dump(point_dict, f)


    def get_clusters_directory(self) -> Path:
        reaction1, reaction2 = self.modelclass.analyze.analyzed_reactions
        delta = self.modelclass.grid.delta
        n_clusters = self.modelclass.clustering.initial_n_clusters
        filename = f"{reaction1}_{reaction2}_clusters_Delta{delta}_NC{n_clusters}.pkl"

        directory = self.get_directory("clustering")

        return directory / filename


    def are_clusters_saved(self):
        return self.get_clusters_directory().exists()
    

    def load_clusters(self) -> tuple | None:
        if not self.are_clusters_saved():
            return None 
        
        path = self.get_clusters_directory()

        with open(path, 'rb') as f:
            clusters_tuple = pickle.load(f)
            return clusters_tuple

    
    def save_clusters(self, n_clusters: int, clusters: np.ndarray) -> None:
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
        delta = self.modelclass.grid.delta
        file = f"{type}_NR{reaction_len}_Delta{delta}"
        if metric_list is not None:
            file += f"_M{len(metric_list)}"

        path = self.dataframe_directory / f"{file}.csv"
        if path.exists() and not overwrite:
            print(f"Skipping existing file: {path.name}")
            return
        
        df.to_csv(path, index=index, encoding='utf-8')

    import xlsxwriter
    def merge_to_excel(self, df_dict: dict[str, pd.DataFrame]) -> None:
        path = Path("result_files") / "DM" / "supplementary" / f"{self.analyzed_tuple_string}.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for df_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=df_name, index=False)


    # =================================================== PLOT DIRECTORY ===================================================


    def save_plot_path(self) -> Path | None:
        reaction1, reaction2 = self.modelclass.analyze.analyzed_reactions
        delta = self.modelclass.grid.delta
        n_clusters = self.modelclass.clustering.grid_n_clusters
        path = self.plots_directory / f"{reaction1}_{reaction2}_NC{n_clusters}_Delta{delta}.png"
        if path.exists():
            return 
        return path