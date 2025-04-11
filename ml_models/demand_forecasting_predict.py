import pandas as pd
import pickle
import os

def run_demand_prediction(return_df=False):
    # Load model and encoders
    with open("ml_models/model.pkl", "rb") as f:
        saved = pickle.load(f)

    model = saved['model']
    encoders = saved['label_encoders']

    # Load and preprocess data
    df = pd.read_csv("data/demand_forecasting.csv")
    df.dropna(inplace=True)

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    features = ['Store ID', 'Price', 'Promotions', 'Seasonality Factors', 'External Factors',
                'Demand Trend', 'Customer Segments', 'Day', 'Month', 'Year']
    X = df[features]

    df['Predicted Sales Quantity'] = model.predict(X)

    # Post-processing
    df['High Demand'] = df['Predicted Sales Quantity'] > df['Predicted Sales Quantity'].quantile(0.75)
    df.sort_values(by='Predicted Sales Quantity', ascending=False, inplace=True)
    
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/predicted_demand.csv", index=False)
    print("✅ Predictions saved to output/predicted_demand.csv")

    if return_df:
        return df

def run_demand_prediction_return_df():
    # Load and preprocess demand data
    df = pd.read_csv("data/demand_forecasting.csv")
    # (Apply any model inference steps here...)

    # Sample result
    df["Predicted Demand"] = df["Sales Quantity"].rolling(window=2, min_periods=1).mean()
    result_df = df[["Product ID", "Store ID", "Predicted Demand"]]

    # Save for compatibility
    os.makedirs("output", exist_ok=True)
    result_df.to_csv("output/predicted_demand.csv", index=False)

    return result_df
