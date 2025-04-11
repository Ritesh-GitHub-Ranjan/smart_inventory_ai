import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import shap
import pickle
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

class DemandForecaster:
    def __init__(self):
        self.models = {}
        self.feature_importances = {}
        self.scaler = None
        
    def _preprocess_data(self, df):
        """Enhanced preprocessing with more features"""
        # Convert date and extract features
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        
        # Add holiday flags
        import holidays
        country_holidays = holidays.CountryHoliday('US')  # Configurable
        df['is_holiday'] = df['Date'].apply(lambda x: x in country_holidays).astype(int)
        
        # Enhanced promotion features
        if 'Promotion Budget' in df.columns:
            df['promo_intensity'] = df['Promotion Budget'] / df['Promotion Budget'].max()
        if 'Promotion Days' in df.columns:
            df['promo_duration_effect'] = df['Promotion Days'] * 0.1
            
        # External indicators
        if 'Weather Index' in df.columns:
            df['weather_impact'] = df['Weather Index'] * 0.05
        if 'Economic Index' in df.columns:
            df['economic_impact'] = (df['Economic Index'] - 100) * 0.1
            
        # Interaction terms
        df['promo_season_interaction'] = df['Promotions'] * df['Seasonality']
        if 'is_high_value' in df.columns and 'weather_impact' in df.columns:
            df['value_weather_interaction'] = df['is_high_value'] * df['weather_impact']
        
        # Add high-value product flag if unit price exists
        if 'Unit Price' in df.columns:
            df['is_high_value'] = (df['Unit Price'] > df['Unit Price'].quantile(0.8)).astype(int)
        
        # Enhanced categorical encoding
        df['Promotions'] = df['Promotions'].map({'Yes': 1, 'No': 0})
        season_map = {'Festival': 2, 'Holiday': 1, 'None': 0}
        # Ensure Seasonality Factors exists before mapping
        if 'Seasonality Factors' in df.columns:
            df['Seasonality'] = df['Seasonality Factors'].map(season_map)
        else:
            df['Seasonality'] = 0  # Default value if column missing
        trend_map = {'Increasing': 1, 'Stable': 0, 'Decreasing': -1}
        df['Demand Trend'] = df['Demand Trend'].map(trend_map)
        
        # One-hot encode customer segments
        df = pd.get_dummies(df, columns=['Customer Segments'], prefix='segment')
        
        # Enhanced external factors handling
        if 'External Factors' in df.columns:
            df['has_competitor'] = df['External Factors'].str.contains('Competitor').astype(int)
            df['bad_weather'] = df['External Factors'].str.contains('Weather').astype(int)
            df['economic_downturn'] = df['External Factors'].str.contains('Economic').astype(int)
        
        # Lag features for time series
        for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month lags
            df[f'lag_{lag}'] = df['Sales Quantity'].shift(lag)
        
        # Rolling statistics
        df['rolling_7_mean'] = df['Sales Quantity'].rolling(7).mean()
        df['rolling_30_mean'] = df['Sales Quantity'].rolling(30).mean()
        
        # Keep Date column for Prophet model while removing others
        return df.drop(columns=['Seasonality Factors', 'External Factors']).dropna()

    def _train_prophet(self, df):
        """Train Prophet model for baseline forecasting"""
        prophet_df = df[['Date', 'Sales Quantity']].rename(columns={
            'Date': 'ds',
            'Sales Quantity': 'y'
        })
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonality for festivals/holidays
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(prophet_df)
        return model

    def _train_xgboost(self, X, y):
        """Train XGBoost with enhanced parameters and validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Check if high-value products exist in data
        has_high_value = 'is_high_value' in X.columns
        
        # Custom objective function for high-value products
        def weighted_squared_error(preds, dtrain):
            labels = dtrain.get_label()
            weights = np.where(X['is_high_value'] == 1, 2.0, 1.0) if has_high_value else np.ones_like(labels)
            grad = (preds - labels) * weights
            hess = np.ones_like(preds) * weights
            return grad, hess
            
        model = xgb.XGBRegressor(
            objective=weighted_squared_error if has_high_value else 'reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric=['rmse', 'mae'],
            random_state=42
        )
        
        # Time-series cross validation
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        return model

    def _train_lightgbm(self, X, y):
        """Train LightGBM model optimized for categorical features"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Check if high-value products exist in data
        has_high_value = 'is_high_value' in X.columns
        
        # Custom objective function for high-value products
        def weighted_mse(y_true, y_pred):
            weights = np.where(X['is_high_value'] == 1, 2.0, 1.0) if has_high_value else np.ones_like(y_true)
            return 'weighted_mse', np.mean(weights * (y_true - y_pred)**2), False
            
        model = lgb.LGBMRegressor(
            objective=weighted_mse if has_high_value else 'regression',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=-1,  # No limit
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_round=50,
            random_state=42
        )
        
        # Time-series cross validation
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['l1', 'l2'],
                categorical_feature=['Seasonality', 'Demand Trend'],
                verbose=-1
            )
        
        return model

    def train_ensemble(self, df, save_path="ml_models/ensemble_model.pkl"):
        """Train ensemble model and save"""
        try:
            # Preprocess data
            df = self._preprocess_data(df)
            
            # Train Prophet model
            self.models['prophet'] = self._train_prophet(df)
            
            # Prepare features for XGBoost
            X = df.drop(columns=['Sales Quantity'])
            y = df['Sales Quantity']
            
            # Train XGBoost
            self.models['xgboost'] = self._train_xgboost(X, y)
            
            # Train LightGBM
            self.models['lightgbm'] = self._train_lightgbm(X, y)
            
            # Calculate feature importance
            self._calculate_feature_importance(X)
            
            # Save model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
                
            logging.info(f"Ensemble model trained and saved to {save_path}")
            return self
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise

    def _calculate_feature_importance(self, X):
        """Calculate and store feature importance metrics"""
        # XGBoost native importance
        self.feature_importances['xgboost'] = {
            'gain': self.models['xgboost'].get_booster().get_score(importance_type='gain'),
            'cover': self.models['xgboost'].get_booster().get_score(importance_type='cover')
        }
        
        # SHAP values
        explainer = shap.Explainer(self.models['xgboost'])
        shap_values = explainer(X)
        self.feature_importances['shap'] = {
            'values': shap_values.values,
            'base_values': shap_values.base_values,
            'data': shap_values.data
        }

    def predict(self, df, return_df=False):
        """Make predictions using ensemble model"""
        try:
            # Load model if not already loaded
            if not self.models:
                with open("ml_models/ensemble_model.pkl", "rb") as f:
                    self = pickle.load(f)
            
            # Preprocess data
            df = self._preprocess_data(df)
            X = df.drop(columns=['Sales Quantity'], errors='ignore')
            
            # Prophet predictions
            prophet_df = df[['Date']].rename(columns={'Date': 'ds'})
            prophet_pred = self.models['prophet'].predict(prophet_df)['yhat'].values
            
            # XGBoost predictions
            xgb_pred = self.models['xgboost'].predict(X)
            
            # LightGBM predictions
            lgb_pred = self.models['lightgbm'].predict(X)
            
            # Weighted ensemble prediction (40% Prophet, 30% XGBoost, 30% LightGBM)
            df['prediction'] = (prophet_pred * 0.4) + (xgb_pred * 0.3) + (lgb_pred * 0.3)
            
            # Calculate prediction intervals
            df['pred_lower'] = df['prediction'] * 0.9  # 10% lower bound
            df['pred_upper'] = df['prediction'] * 1.1  # 10% upper bound
            
            return df if return_df else df[['prediction', 'pred_lower', 'pred_upper']]
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

    def evaluate(self, df):
        """Evaluate model with business metrics"""
        result_df = self.predict(df, return_df=True)
        y_true = result_df['Sales Quantity']
        y_pred = result_df['prediction']
        
        # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
        
        # Calculate quantile coverage
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_coverage = {}
        for q in quantiles:
            lower = y_pred * (1 - q)
            upper = y_pred * (1 + q)
            coverage = ((y_true >= lower) & (y_true <= upper)).mean()
            quantile_coverage[f'coverage_{int(q*100)}'] = coverage
            
        # High-value product metrics if available
        high_value_metrics = {}
        if 'is_high_value' in result_df.columns:
            high_value_mask = result_df['is_high_value'] == 1
            high_value_metrics = {
                'high_value_MAE': mean_absolute_error(y_true[high_value_mask], y_pred[high_value_mask]),
                'high_value_WMAPE': np.sum(np.abs(y_true[high_value_mask] - y_pred[high_value_mask])) / 
                                   np.sum(y_true[high_value_mask])
            }
        
        # Business impact calculations if unit price exists
        business_impact = {}
        if 'Unit Price' in result_df.columns:
            # Calculate revenue impact
            revenue_loss = np.sum((y_true - y_pred).clip(lower=0) * result_df['Unit Price'])
            revenue_excess = np.sum((y_pred - y_true).clip(lower=0) * result_df['Unit Price'])
            
            # Calculate inventory cost impact (assuming 20% holding cost)
            holding_cost = 0.2
            excess_inventory_cost = np.sum((y_pred - y_true).clip(lower=0) * result_df['Unit Price'] * holding_cost)
            
            business_impact = {
                'revenue_loss': revenue_loss,
                'revenue_excess': revenue_excess,
                'excess_inventory_cost': excess_inventory_cost,
                'total_business_impact': revenue_loss + revenue_excess + excess_inventory_cost
            }
        
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'WMAPE': wmape,
            'Bias': (y_pred - y_true).mean(),
            **quantile_coverage,
            **high_value_metrics,
            **business_impact
        }
        
        return metrics

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load dataset
        df = pd.read_csv("data/demand_forecasting.csv")
        
        # Train and save model
        forecaster = DemandForecaster()
        forecaster.train_ensemble(df)
        
        # Evaluate model
        metrics = forecaster.evaluate(df)
        logging.info(f"Model evaluation metrics: {metrics}")
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
