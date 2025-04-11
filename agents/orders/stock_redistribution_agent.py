import pandas as pd
import os

class StockRedistribution:
    def __init__(self, inventory_df):
        self.inventory_df = inventory_df

    def redistribute_stock(self):
        """Redistribute stock between overstocked and understocked items"""
        try:
            # Validate required columns exist
            required_cols = ['Stock Levels', 'Reorder Point', 'Product ID']
            if not all(col in self.inventory_df.columns for col in required_cols):
                raise ValueError(f"Input DataFrame missing required columns: {required_cols}")

            # Create output directory if needed
            os.makedirs("output", exist_ok=True)

            # Find over/under stocked items
            overstocked = self.inventory_df[self.inventory_df['Stock Levels'] > self.inventory_df['Reorder Point']]
            understocked = self.inventory_df[self.inventory_df['Stock Levels'] < self.inventory_df['Reorder Point']]

            # Track if any redistribution occurred
            redistributed = False

            for idx, overstock_row in overstocked.iterrows():
                for jdx, understock_row in understocked.iterrows():
                    if overstock_row['Product ID'] == understock_row['Product ID']:
                        transfer_quantity = min(
                            overstock_row['Stock Levels'] - overstock_row['Reorder Point'],
                            understock_row['Reorder Point'] - understock_row['Stock Levels']
                        )
                        if transfer_quantity > 0:
                            self.inventory_df.loc[idx, 'Stock Levels'] -= transfer_quantity
                            self.inventory_df.loc[jdx, 'Stock Levels'] += transfer_quantity
                            redistributed = True

            # Always save the result, even if no redistribution occurred
            self.inventory_df.to_csv('output/redistributed_inventory.csv', index=False)
            
            return {
                'status': 'success',
                'redistributed': redistributed,
                'output_file': 'output/redistributed_inventory.csv'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
