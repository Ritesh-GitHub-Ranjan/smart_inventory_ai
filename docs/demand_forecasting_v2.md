# Demand Forecasting System v2

## Overview
This document describes the enhanced demand forecasting system that combines:
- Prophet for time-series forecasting
- XGBoost for feature-based predictions
- SHAP values for model interpretability

## Key Improvements
1. **Ensemble Modeling**: Combines strengths of both time-series and feature-based approaches
2. **Enhanced Feature Engineering**: Added lag features, rolling statistics, and better categorical encoding
3. **Business Metrics**: Tracks WMAPE, forecast bias, and coverage in addition to RMSE
4. **Interpretability**: Provides SHAP values and feature importance scores
5. **Robust Validation**: Uses time-series cross validation

## Components

### 1. Core Model (`demand_forecasting_v2.py`)
- Implements the ensemble forecasting logic
- Handles data preprocessing and feature engineering
- Manages model training and evaluation

### 2. Forecasting Agent (`forecasting_agent_v2.py`)
- Provides interface for the forecasting system
- Handles model loading and prediction
- Manages feature importance analysis

### 3. Test Suite
- Unit tests for model components (`test_demand_forecasting_v2.py`)
- Integration tests for agent (`test_forecasting_agent_v2.py`)

## Implementation Steps

### 1. Training the Model
```bash
python ml_models/demand_forecasting_v2.py
```

### 2. Using the Forecasting Agent
```python
from agents.forecasting_agent_v2 import ForecastingAgent

# Initialize agent
agent = ForecastingAgent()

# Make predictions
predictions = agent.predict_demand(input_data)

# Get feature importance
importance = agent.get_feature_importance()
```

### 3. Migration from v1
```bash
python scripts/migrate_to_v2_forecasting.py
```

## Performance Metrics
The system tracks:
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- 90% Prediction Interval Coverage
- Forecast Bias
- Feature Importance Scores

## Monitoring
The system logs:
- Training metrics
- Prediction results
- Feature importance changes over time

## Dependencies
- Python 3.8+
- Required packages:
  - prophet
  - xgboost
  - shap
  - scikit-learn
  - pandas

## Maintenance
To retrain the model:
1. Update the training data in `data/demand_forecasting.csv`
2. Run the training script
3. Verify performance metrics
4. Deploy the new model

## Troubleshooting
Common issues:
- **Missing data**: Ensure all required features are present
- **Model performance degradation**: Check feature distributions for drift
- **Prediction errors**: Verify input data matches training schema
