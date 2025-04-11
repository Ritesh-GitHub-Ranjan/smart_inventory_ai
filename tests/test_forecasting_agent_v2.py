import pytest
import pandas as pd
import numpy as np
from archive.forecasting_agent_v2 import ForecastingAgent
import os
import tempfile
import pickle
from unittest.mock import patch

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=30)
    return pd.DataFrame({
        'Date': dates,
        'Product ID': ['P1001'] * len(dates),
        'Sales Quantity': np.random.poisson(50, len(dates)),
        'Price': np.random.normal(9.99, 1.5, len(dates)),
        'Promotions': ['Yes' if x > 0.7 else 'No' for x in np.random.rand(len(dates))],
        'Seasonality Factors': ['None'] * len(dates),
        'External Factors': ['None'] * len(dates),
        'Demand Trend': ['Stable'] * len(dates),
        'Customer Segments': ['Regular'] * len(dates)
    })

@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)

def test_agent_initialization(temp_model_file):
    """Test that agent initializes correctly"""
    agent = ForecastingAgent(model_path=temp_model_file)
    assert agent is not None
    assert agent.forecaster is None
    assert agent.model_path == temp_model_file

def test_model_training(sample_data, temp_model_file):
    """Test that agent can train and save model"""
    with patch('pandas.read_csv', return_value=sample_data):
        agent = ForecastingAgent(model_path=temp_model_file)
        result = agent.train_model()
        
        assert result['status'] == 'success'
        assert 'metrics' in result
        assert os.path.exists(temp_model_file)
        
        # Verify model can be loaded
        with open(temp_model_file, 'rb') as f:
            model = pickle.load(f)
            assert model is not None

def test_prediction(sample_data, temp_model_file):
    """Test that agent can make predictions"""
    # First train a model
    with patch('pandas.read_csv', return_value=sample_data):
        agent = ForecastingAgent(model_path=temp_model_file)
        agent.train_model()
    
    # Test prediction
    prediction_result = agent.predict_demand(sample_data.head(5))
    assert prediction_result['status'] == 'success'
    assert 'predictions' in prediction_result
    assert len(prediction_result['predictions']) == 5
    assert all(k in prediction_result['predictions'][0] 
               for k in ['prediction', 'pred_lower', 'pred_upper'])

def test_feature_importance(sample_data, temp_model_file):
    """Test that agent can provide feature importance"""
    # First train a model
    with patch('pandas.read_csv', return_value=sample_data):
        agent = ForecastingAgent(model_path=temp_model_file)
        agent.train_model()
    
    # Test feature importance
    importance_result = agent.get_feature_importance()
    assert importance_result['status'] == 'success'
    assert 'feature_importance' in importance_result
    assert 'xgboost' in importance_result['feature_importance']
    assert 'shap' in importance_result['feature_importance']

def test_error_handling(sample_data, temp_model_file):
    """Test error handling for invalid inputs"""
    agent = ForecastingAgent(model_path=temp_model_file)
    
    # Test with invalid data
    with pytest.raises(Exception):
        agent.predict_demand(pd.DataFrame())
    
    # Test feature importance before training
    if os.path.exists(temp_model_file):
        os.unlink(temp_model_file)
    importance_result = agent.get_feature_importance()
    assert importance_result['status'] == 'error'

def test_model_loading(sample_data, temp_model_file):
    """Test that agent loads model correctly"""
    # First train and save a model
    with patch('pandas.read_csv', return_value=sample_data):
        agent = ForecastingAgent(model_path=temp_model_file)
        agent.train_model()
    
    # Create new agent instance and test loading
    new_agent = ForecastingAgent(model_path=temp_model_file)
    assert new_agent.forecaster is not None
    
    # Verify prediction works
    result = new_agent.predict_demand(sample_data.head(1))
    assert result['status'] == 'success'
