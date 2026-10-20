"""Register every model here, one line each. `python run.py models` runs them all.

Owners: Chamathka adds her ts models, Teddy adds his ml models. Open a pull request.
"""
from src.models.baselines import MeanForecaster, PersistenceForecaster, ZeroForecaster
from src.models.ml.logistic import LogisticForecaster
from src.models.ml.random_forest import RandomForestForecaster
from src.models.ml.xgboost_model import XGBoostForecaster
from src.models.template_linear import TemplateAR5
from src.models.ts.arma import ARForecaster, ARMAForecaster
from src.models.ts.arx import StockARXForecaster
from src.models.ts.garch import GarchForecaster

MODELS = {
    "zero": ZeroForecaster,
    "hist_mean": MeanForecaster,
    "persistence": PersistenceForecaster,
    "template_ar5": TemplateAR5,      # TEMPLATE: remove once the real AR/ARMA exist
    "ar": ARForecaster,             # Chamathka, src/models/ts/arma.py
    "arma": ARMAForecaster,
    "ar_gjr_garch_t": GarchForecaster,   # src/models/ts/garch.py
    "stock_arx": StockARXForecaster,
    "logistic": LogisticForecaster,      # Teddy, src/models/ml/
    "random_forest": RandomForestForecaster,
    "xgboost": XGBoostForecaster,
    # "lstm": LSTMForecaster,
}
