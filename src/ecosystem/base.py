from cobra import Model

# local scripts
from .community import EcosystemCommunity
from .grid import EcosystemGrid
from .analyze import EcosystemAnalyze
from .plot import EcosystemPlot
from scripts.model_clustering import ModelClustering
from scripts.model_io import ModelIO

class BaseEcosystem():
    """Abstract base class for ecosystem models

    This class is meant to be inherited and not to be used on its own. 
    Holds all attributes and methods in common between FullEcosystem and PrecomputedEcosystem."""

    def __init__(self, community_name: str = "community", community_id: str = "community"):
        self.community_name = community_name                
        self.community_id = community_id
                                 
        self.community_model: Model = Model(id_or_model=community_id, name=community_name)
        self.size: int  = 0   
        self.objectives: list[dict[str, float]] = [] 

        self.member_model_ids: list[str] = []
        
        self._build_modules()
        
        
    def _build_modules(self):
        self.community = EcosystemCommunity(self)
        self.grid = EcosystemGrid(self)
        self.analyze = EcosystemAnalyze(self)
        self.plot = EcosystemPlot(self)
        self.clustering = ModelClustering(self)
        self.io = ModelIO(self, self.community_name)


