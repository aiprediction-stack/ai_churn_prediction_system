# services/model_interface.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class ModelInterface(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        pass