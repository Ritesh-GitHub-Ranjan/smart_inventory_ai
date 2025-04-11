import pandas as pd
from ml_models.demand_forecasting_v2 import DemandForecaster
import logging
from typing import Dict, Any
import json
import os
import pickle
import numpy as np


class ForecastingAgent:
    def __init__(self, model_path: str = "ml_models/ensemble_model.pkl"):
        self.model_path = model_path
        self.forecaster = None
        self.logger = logging.getLogger(__name__)
        self._load_model()

    def _load_model(self):
        """Load the trained forecasting model"""
        try:
            if os.path.exists(self.model_path):
                self.forecaster = DemandForecaster()
                with open(self.model_path, 'rb') as f:
                    self.forecaster = pickle.load(f)
                self.logger.info("Forecasting model loaded successfully")
            else:
                self.logger.warning("No trained model found at %s", self.model_path)
        except Exception as e:
            self.logger.error("Failed to load forecasting model: %s", str(e))
            raise

    def train_model(self, data_path: str = "data/demand_forecasting.csv") -> Dict[str, Any]:
        """Train and save the forecasting model"""
        try:
            df = pd.read_csv(data_path)
            self.forecaster = DemandForecaster()
            self.forecaster.train_ensemble(df, save_path=self.model_path)
            
            # Evaluate model performance
            metrics = self.forecaster.evaluate(df)
            self.logger.info("Model training completed with metrics: %s", metrics)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'model_path': self.model_path
            }
        except Exception as e:
            self.logger.error("Model training failed: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }

    def predict_demand(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make demand predictions for given input data"""
        try:
            if self.forecaster is None:
                self._load_model()
            
            predictions = self.forecaster.predict(input_data, return_df=True)
            
            # Format results for API response
            results = predictions[['prediction', 'pred_lower', 'pred_upper']].to_dict('records')
            
            return {
                'status': 'success',
                'predictions': results,
                'features': list(input_data.columns)
            }
        except Exception as e:
            self.logger.error("Prediction failed: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance analysis"""
        try:
            if self.forecaster is None:
                self._load_model()
            
            return {
                'status': 'success',
                'feature_importance': {
                    'xgboost': self.forecaster.feature_importances['xgboost'],
                    'shap': {
                        'average_abs_shap': np.abs(self.forecaster.feature_importances['shap']['values']).mean(axis=0).tolist(),
                        'feature_names': list(self.forecaster.feature_importances['shap']['data'].columns)
                    }
                }
            }
        except Exception as e:
            self.logger.error("Feature importance analysis failed: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }

    def run_backtest(self, periods: int = 12) -> Dict[str, Any]:
        """Run backtesting on historical data"""
        try:
            if self.forecaster is None:
                self._load_model()
            
            df = pd.read_csv("data/demand_forecasting.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Split data into training and test sets
            test_size = int(len(df) * 0.2)  # Hold out 20% for testing
            train_df = df.iloc[:-test_size]
            test_df = df.iloc[-test_size:]
            
            # Train on training data
            self.forecaster.train_ensemble(train_df)
            
            # Evaluate on test data
            metrics = self.forecaster.evaluate(test_df)
            
            return {
                'status': 'success',
                'backtest_metrics': metrics,
                'periods': periods,
                'test_size': test_size
            }
        except Exception as e:
            self.logger.error("Backtesting failed: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        try:
            if self.forecaster is None:
                self._load_model()
            
            df = pd.read_csv("data/demand_forecasting.csv")
            metrics = self.forecaster.evaluate(df)
            
            return {
                'status': 'success',
                'performance_metrics': metrics
            }
        except Exception as e:
            self.logger.error("Performance monitoring failed: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }

# Integrating with ReorderAgent
class ReorderAgent:
    def __init__(self):
        pass
    
    def calculate_reorder(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate reorder points based on demand forecast"""
        reorder_points = forecast_df.copy()
        reorder_points['reorder_point'] = reorder_points['prediction'] * 0.9  # Mock reorder logic
        return reorder_points


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agents
    forecasting_agent = ForecastingAgent()
    reorder_agent = ReorderAgent()

    # Train model (if not already trained)
    if not os.path.exists(forecasting_agent.model_path):
        training_result = forecasting_agent.train_model()
        print("Training result:", training_result)
    
    # Load sample data for prediction
    sample_data = pd.read_csv("data/demand_forecasting.csv").head(10)
    
    # Make predictions using ForecastingAgent
    prediction_result = forecasting_agent.predict_demand(sample_data)
    print("Prediction result:", json.dumps(prediction_result, indent=2))
    
    # Pass predictions to ReorderAgent for reorder calculation
    forecast_df = pd.DataFrame(prediction_result['predictions'])  # Prepare data for reorder
    reorder_result = reorder_agent.calculate_reorder(forecast_df)
    print("Reorder result:", reorder_result)
