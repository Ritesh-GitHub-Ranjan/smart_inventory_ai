from prophet import Prophet
import pandas as pd
from prophet.make_holidays import make_holidays_df


class DemandAgent:
    def __init__(self):
        # Configure holidays and seasonality
        year_list = [2023, 2024, 2025]  # Extend as needed
        holidays = make_holidays_df(year_list=year_list, country='US')
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays,
            seasonality_mode='multiplicative'
        )

    def _prepare_prophet_data(self, df):
        """Prepare data for Prophet forecasting"""
        # Convert to Prophet expected format
        prophet_df = df.rename(columns={
            'Date': 'ds',
            'Sales Quantity': 'y'
        })
        
        # Add external regressors
        for col in ['Promotions', 'Seasonality', 'Demand Trend']:
            if col in df.columns:
                prophet_df[col] = df[col]
                self.model.add_regressor(col)
        
        # Add customer segments as regressors
        for col in df.columns:
            if col.startswith('Customer Segments_'):
                prophet_df[col] = df[col]
                self.model.add_regressor(col)
                
        return prophet_df[['ds', 'y'] + [c for c in prophet_df.columns if c not in ['ds', 'y']]]

    def train_demand_forecasting(self, demand_df):
        """Train Prophet model with additional features"""
        try:
            # Prepare data with all features
            prophet_df = self._prepare_prophet_data(demand_df)
            self.model.fit(prophet_df)
        except Exception as e:
            raise Exception(f"Prophet training failed: {str(e)}")

    def predict_future_demand(self, periods=30, feature_values=None):
        """Predict future demand with optional feature values"""
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Add default feature values if not provided
            if feature_values is None:
                feature_values = {
                    'Promotions': 0,
                    'Seasonality': 0,
                    'Demand Trend': 0
                }
                # Add default customer segments
                for col in future.columns:
                    if col.startswith('Customer Segments_'):
                        feature_values[col] = 0
            
            # Apply feature values to future dataframe
            for col, val in feature_values.items():
                future[col] = val
                
            forecast = self.model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    def analyze_demand(self, demand_df, periods=30, feature_values=None):
        """Complete demand analysis with forecast"""
        self.train_demand_forecasting(demand_df)
        return self.predict_future_demand(periods, feature_values)
