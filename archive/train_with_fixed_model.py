import pandas as pd
from ml_models.demand_forecasting_v2_fixed import DemandForecasterFixed
import logging
import traceback

def main():
    try:
        # Load data
        df = pd.read_csv('data/demand_forecasting.csv')
        
        # Initialize and train model
        forecaster = DemandForecasterFixed()
        model = forecaster.train_ensemble(df)
        print('Model trained successfully')
        return True
    except Exception as e:
        print(f'Error: {str(e)}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
