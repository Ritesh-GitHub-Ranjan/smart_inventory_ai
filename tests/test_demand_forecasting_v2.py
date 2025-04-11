import pytest
import pandas as pd
import numpy as np
from ml_models.demand_forecasting_v2 import DemandForecaster
import os
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
    """Generate realistic sample data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31')
    np.random.seed(42)
    
    data = {
        'Date': dates,
        'Product ID': ['P1001'] * len(dates),
        'Store ID': ['S001'] * len(dates),
        'Sales Quantity': np.random.poisson(50, len(dates)) + 
                         (dates.dayofweek == 5) * 20 +  # Higher on Saturdays
                         (dates.day == 1) * 30,         # Higher on 1st of month
        'Price': np.random.normal(9.99, 1.5, len(dates)),
        'Promotions': ['Yes' if x > 0.7 else 'No' for x in np.random.rand(len(dates))],
        'Seasonality Factors': ['None'] * len(dates),
        'External Factors': ['None'] * len(dates),
        'Demand Trend': ['Stable'] * len(dates),
        'Customer Segments': ['Regular'] * len(dates)
    }
    
    # Add some seasonality
    data['Seasonality Factors'][15] = 'Holiday'
    data['Sales Quantity'][15] += 40
    
    return pd.DataFrame(data)

def test_model_training(sample_data):
    """Test that the model can be trained successfully"""
    forecaster = DemandForecaster()
    model = forecaster.train_ensemble(sample_data)
    assert model is not None
    assert 'prophet' in model.models
    assert 'xgboost' in model.models
    assert os.path.exists('ml_models/ensemble_model.pkl')

def test_model_prediction(sample_data):
    """Test that the model can make predictions"""
    forecaster = DemandForecaster()
    forecaster.train_ensemble(sample_data)
    predictions = forecaster.predict(sample_data)
    assert not predictions.empty
    assert 'prediction' in predictions.columns
    assert 'pred_lower' in predictions.columns
    assert 'pred_upper' in predictions.columns

def test_model_evaluation(sample_data):
    """Test that evaluation metrics are calculated correctly"""
    forecaster = DemandForecaster()
    forecaster.train_ensemble(sample_data)
    metrics = forecaster.evaluate(sample_data)
    assert isinstance(metrics, dict)
    assert 'MAE' in metrics
    assert 'MAPE' in metrics
    assert 'Coverage_90' in metrics
    assert 'Bias' in metrics
    assert 0 <= metrics['MAPE'] <= 100  # MAPE should be percentage

def test_feature_importance(sample_data):
    """Test that feature importance is calculated"""
    forecaster = DemandForecaster()
    forecaster.train_ensemble(sample_data)
    assert 'xgboost' in forecaster.feature_importances
    assert 'shap' in forecaster.feature_importances
    assert len(forecaster.feature_importances['xgboost']['gain']) > 0

def test_preprocessing(sample_data):
    """Test that preprocessing creates expected features"""
    forecaster = DemandForecaster()
    processed = forecaster._preprocess_data(sample_data)
    assert 'day_of_week' in processed.columns
    assert 'month' in processed.columns
    assert 'Promotions' in processed.columns
    assert processed['Promotions'].isin([0, 1]).all()
    assert 'lag_7' in processed.columns
    assert 'rolling_7_mean' in processed.columns

def test_time_series_validation(sample_data):
    """Test that time series validation works"""
    forecaster = DemandForecaster()
    processed = forecaster._preprocess_data(sample_data)
    X = processed.drop(columns=['Sales Quantity'])
    y = processed['Sales Quantity']
    
    model = forecaster._train_xgboost(X, y)
    assert model is not None
    assert model.n_features_in_ == X.shape[1]
