import pandas as pd
import os
import logging

class SalesImpactAgent:
    def __init__(self, pricing_path="data/pricing_optimization.csv", forecast_path="output/predicted_demand.csv"):
        self.pricing_path = pricing_path
        self.forecast_path = forecast_path
        self.output_path = "output/sales_impact_report.csv"

        self.logger = logging.getLogger('SalesImpactAgent')
        handler = logging.FileHandler('logs/sales_impact_agent.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self):
        self.logger.info("Running SalesImpactAgent...")

        if not os.path.exists(self.pricing_path) or not os.path.exists(self.forecast_path):
            self.logger.error(" Required files not found.")
            print(" Required files not found.")
            return

        pricing_df = pd.read_csv(self.pricing_path)
        forecast_df = pd.read_csv(self.forecast_path)

        # Clean column names
        pricing_df.columns = pricing_df.columns.str.strip()
        forecast_df.columns = forecast_df.columns.str.strip()

        self.logger.info(f"Loaded {len(pricing_df)} pricing records and {len(forecast_df)} forecast records")

        # Merge with suffixes to avoid column name conflict
        merged = pd.merge(
            pricing_df,
            forecast_df,
            on=["Product ID", "Store ID"],
            how="inner",
            suffixes=('_pricing', '_forecast')
        )

        
        # print("\n Merged Columns:\n", merged.columns.tolist())  # 👈 Debug print

        # Check column names and use correct ones
        try:
            price_col = 'Price_pricing'
            predicted_col = 'Predicted Sales Quantity'

            merged["Price Change %"] = ((merged[price_col] - merged["Competitor Prices"]) / merged["Competitor Prices"]) * 100
            merged["Sales Lift %"] = ((merged["Sales Volume"] - merged[predicted_col]) / merged[predicted_col]) * 100
        except KeyError as e:
            self.logger.error(f" Missing column: {e}")
            print(f" Column not found: {e}")
            print(" Available columns:", merged.columns.tolist())
            return

        # Classify impact
        def classify(row):
            if row["Sales Lift %"] > 10:
                return " Effective"
            elif row["Sales Lift %"] < -10:
                return " Ineffective"
            else:
                return " Neutral"

        merged["Impact"] = merged.apply(classify, axis=1)

        merged.to_csv(self.output_path, index=False)
        self.logger.info(f" Sales Impact Analysis completed with {len(merged)} merged records")
        print(f" Sales Impact Analysis completed with {len(merged)} records.")
