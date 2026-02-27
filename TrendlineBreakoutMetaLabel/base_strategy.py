from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    Toute stratégie qui veut profiter du Meta-Labeling (IA XGBoost)
    doit implémenter la méthode 'generate_dataset'.
    """

    @abstractmethod
    def generate_dataset(self, ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Analyse les données OHLCV et retourne les trades bruts trouvés par la logique technique "bête" (sans IA),
        les données contextuelles (features) pour entraîner l'IA, et les labels binaires (succès ou non).
        
        Args:
            ohlcv (pd.DataFrame): Données de marché (doit contenir 'open', 'high', 'low', 'close', 'volume')
            
        Returns:
            tuple:
                - trades (pd.DataFrame): Métadonnées des trades détectés (entry_i, entry_p, exit_i, exit_p, return, etc)
                - data_x (pd.DataFrame): Features du Machine Learning (indicateurs techniques, positions, etc)
                - data_y (pd.Series):    Labels binaires d'apprentissage (1 = Profit, 0 = Perte/SL)
        """
        pass
