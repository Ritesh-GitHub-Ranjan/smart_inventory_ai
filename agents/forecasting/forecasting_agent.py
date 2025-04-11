import pandas as pd
import numpy as np
import os
import pickle
import logging
from functools import wraps
from typing import Dict, Any

from ml_models.demand_forecasting_v2 import DemandForecaster

# ------------------ Decorator ------------------

def ensure_model_loaded(func):
    """Decorator to ensure model is loaded before any method runs"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.forecaster is None:
            self._load_model()
        return func(self, *args, **kwargs)
    return wrapper

# ------------------ Utility Functions ------------------

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-specific preprocessing"""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df['Promotions'] = df['Promotions'].map({'Yes': 1, 'No': 0})
    df['Seasonality'] = df['Seasonality Factors'].map({
        'Festival': 2,
        'Holiday': 1,
        'None': 0
    })
    df['Demand Trend'] = df['Demand Trend'].map({
        'Increasing': 1,
        'Stable': 0,
        'Decreasing': -1
    })

    df = pd.get_dummies(df, columns=['Customer Segments'], drop_first=True)

    return df

def run_demand_prediction(model: DemandForecaster, input_df: pd.DataFrame) -> pd.DataFrame:
    """Runs prediction with preprocessing and returns output DataFrame"""
    processed_df = preprocess_input(input_df.copy())
    return model.predict(processed_df, return_df=True)

# ------------------ ForecastingAgent ------------------

class ForecastingAgent:
    def __init__(self, model_path: str = "ml_models/ensemble_model.pkl"):
        self.model_path = model_path
        self.forecaster = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self._load_model()

    def _load_model(self):
        """Load the forecasting model"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.forecaster = pickle.load(f)
            self.logger.info("Forecasting model loaded successfully")
        else:
            self.forecaster = DemandForecaster()
            self.logger.warning("No trained model found, initialized a new forecaster")

    def train_model(self, data_path: str = "data/demand_forecasting.csv") -> Dict[str, Any]:
        try:
            df = pd.read_csv(data_path)
            df = preprocess_input(df)

            self.forecaster = DemandForecaster()
            self.forecaster.train_ensemble(df, save_path=self.model_path)

            metrics = self.forecaster.evaluate(df)
            return {'status': 'success', 'metrics': metrics, 'model_path': self.model_path}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    @ensure_model_loaded
    def predict_demand(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            predictions = run_demand_prediction(self.forecaster, input_data)
            return {
                'status': 'success',
                'predictions': predictions[['prediction', 'pred_lower', 'pred_upper']].to_dict('records'),
                'features': list(input_data.columns)
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    @ensure_model_loaded
    def get_feature_importance(self) -> Dict[str, Any]:
        try:
            shap_vals = self.forecaster.feature_importances['shap']
            return {
                'status': 'success',
                'feature_importance': {
                    'xgboost': self.forecaster.feature_importances['xgboost'],
                    'shap': {
                        'average_abs_shap': np.abs(shap_vals['values']).mean(axis=0).tolist(),
                        'feature_names': list(shap_vals['data'].columns)
                    }
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    @ensure_model_loaded
    def run_backtest(self, periods: int = 12) -> Dict[str, Any]:
        try:
            df = pd.read_csv("data/demand_forecasting.csv")
            df = preprocess_input(df)
            df = df.sort_values('Date')

            test_size = int(len(df) * 0.2)
            train_df, test_df = df[:-test_size], df[-test_size:]

            self.forecaster.train_ensemble(train_df)
            metrics = self.forecaster.evaluate(test_df)

            return {
                'status': 'success',
                'backtest_metrics': metrics,
                'periods': periods,
                'test_size': test_size
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    @ensure_model_loaded
    def get_model_performance(self) -> Dict[str, Any]:
        try:
            df = pd.read_csv("data/demand_forecasting.csv")
            df = preprocess_input(df)
            metrics = self.forecaster.evaluate(df)
            return {'status': 'success', 'performance_metrics': metrics}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# ------------------ ReorderAgent (for demo) ------------------

class ReorderAgent:
    def __init__(self): pass

    def calculate_reorder(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        forecast_df['reorder_point'] = forecast_df['prediction'] * 0.9
        return forecast_df

# ------------------ Demo ------------------

if __name__ == "__main__":
    forecasting_agent = ForecastingAgent()
    reorder_agent = ReorderAgent()

    # Train model if needed
    if not os.path.exists(forecasting_agent.model_path):
        print(forecasting_agent.train_model())

    # Predict
    sample_data = pd.read_csv("data/demand_forecasting.csv").head(10)
    prediction_result = forecasting_agent.predict_demand(sample_data)
    print("Predictions:", prediction_result)

    # Reorder logic
    forecast_df = pd.DataFrame(prediction_result['predictions'])
    reorder_df = reorder_agent.calculate_reorder(forecast_df)
    print("Reorder Result:")
    print(reorder_df.head())
