import logging
import pandas as pd
import os

class PricingAgent:
    def __init__(self, data_path='data/pricing_optimization.csv'):
        self.data_path = data_path
        self.logger = logging.getLogger('PricingAgent')
        handler = logging.FileHandler('logs/pricing_agent.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self):
        self.logger.info("Running PricingAgent...")

        try:
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded data from {self.data_path} with shape {df.shape}")

            df['Price-to-Sales Ratio'] = df['Price'] / df['Sales Volume']
            overpriced = df[df['Price-to-Sales Ratio'] > 1.5]
            underpriced = df[df['Price-to-Sales Ratio'] < 0.5]

            os.makedirs("output", exist_ok=True)
            overpriced.to_csv("output/overpriced_products.csv", index=False)
            underpriced.to_csv("output/underpriced_products.csv", index=False)

            self.logger.info(f"Saved {len(overpriced)} overpriced products")
            self.logger.info(f"Saved {len(underpriced)} underpriced products")

            print("Pricing analysis complete")
            print(f"Overpriced products saved to output/overpriced_products.csv")
            print(f"Underpriced products saved to output/underpriced_products.csv")

        except Exception as e:
            self.logger.error(f"Error in PricingAgent: {e}")

        self.logger.info("Finished PricingAgent")
