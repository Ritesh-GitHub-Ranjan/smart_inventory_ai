# demand_forecasting_model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
import os

def preprocess_data(df):
    """Preprocess data for XGBoost forecasting"""
    # Convert date to features
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    
    # Encode categorical features with error handling
    if 'Promotions' in df.columns:
        df['Promotions'] = df['Promotions'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    if 'Seasonality Factors' in df.columns:
        df['Seasonality'] = df['Seasonality Factors'].map({
            'Festival': 2,
            'Holiday': 1,
            'None': 0
        }).fillna(0)
    else:
        df['Seasonality'] = 0
        
    if 'Demand Trend' in df.columns:
        df['Demand Trend'] = df['Demand Trend'].map({
            'Increasing': 1,
            'Stable': 0,
            'Decreasing': -1
        }).fillna(0)
    
    # One-hot encode customer segments
    df = pd.get_dummies(df, columns=['Customer Segments'])
    
    # Handle external factors with error handling
    if 'External Factors' in df.columns:
        df['External_Factor_Competitor'] = df['External Factors'].str.contains('Competitor', na=False).astype(int)
        df['External_Factor_Weather'] = df['External Factors'].str.contains('Weather', na=False).astype(int)
        df['External_Factor_Economic'] = df['External Factors'].str.contains('Economic', na=False).astype(int)
    else:
        df['External_Factor_Competitor'] = 0
        df['External_Factor_Weather'] = 0
        df['External_Factor_Economic'] = 0
    
    return df.drop(columns=['Date', 'Seasonality Factors', 'External Factors'])

def train_model(df, save_path="ml_models/model.pkl"):
    """Train and save XGBoost demand forecasting model"""
    try:
        # Preprocess data
        df = preprocess_data(df)
        
        # Prepare features and target
        X = df.drop(columns=['Sales Quantity'])
        y = df['Sales Quantity']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model with improved parameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric='rmse',
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
            
        logging.info(f"Model trained and saved to {save_path}")
        return model
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def run_demand_prediction(df, return_df=False):
    """Run demand prediction using trained model"""
    try:
        # Load model
        with open("ml_models/model.pkl", "rb") as f:
            model = pickle.load(f)
            
        # Preprocess new data
        df = preprocess_data(df)
        X = df.drop(columns=['Sales Quantity'], errors='ignore')
        
        # Make predictions
        df['prediction'] = model.predict(X)
        
        return df if return_df else df['prediction']
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load dataset
        df = pd.read_csv("data/demand_forecasting.csv")
        df.dropna(inplace=True)
        
        # Train and save model
        train_model(df)
        logging.info("Model training completed successfully")
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
