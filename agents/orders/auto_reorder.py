import pandas as pd
import os
import logging
from typing import Optional, List, Dict
from datetime import datetime


class AutoReorder:
    def __init__(
        self,
        reorder_path: str,
        inventory_path: str,
        output_path: str = "output/auto_reorders.csv"
    ):
        """Initialize AutoReorder with paths to reorder suggestions and inventory data.
        
        Args:
            reorder_path: Path to CSV file containing reorder suggestions
            inventory_path: Path to current inventory CSV file
            output_path: Path to save processed reorder records
        """
        self.reorder_path = reorder_path
        self.inventory_path = inventory_path
        self.output_path = output_path
        
        # Configure logging
        self.logger = logging.getLogger("AutoReorder")
        self.logger.setLevel(logging.INFO)

    def process_reorders(self) -> Optional[pd.DataFrame]:
        """Process automatic reorders and update inventory levels.
        
        Returns:
            DataFrame of processed reorders if successful, None otherwise
        """
        try:
            # Validate input files exist
            if not os.path.exists(self.reorder_path):
                raise FileNotFoundError(f"Reorder file not found: {self.reorder_path}")
            if not os.path.exists(self.inventory_path):
                raise FileNotFoundError(f"Inventory file not found: {self.inventory_path}")

            self.logger.info("Starting auto-reorder processing")
            
            # Load data with Store ID validation
            reorder_df = pd.read_csv(self.reorder_path)
            inventory_df = pd.read_csv(self.inventory_path)
            
            # Validate Store IDs
            if 'Store ID' not in reorder_df.columns:
                raise ValueError("Reorder data missing Store ID column")
            if 'Store ID' not in inventory_df.columns:
                raise ValueError("Inventory data missing Store ID column")
                
            # Convert Store IDs to strings for consistency
            reorder_df['Store ID'] = reorder_df['Store ID'].astype(str)
            inventory_df['Store ID'] = inventory_df['Store ID'].astype(str)
            
            # Log Store ID info
            unique_store_ids = reorder_df['Store ID'].unique()
            self.logger.info(f"Processing reorders for {len(unique_store_ids)} stores: {unique_store_ids}")

            updated_rows = []
            processed_count = 0
            
            # Process each reorder suggestion
            for _, row in reorder_df.iterrows():
                product_id = row['Product ID']
                store_id = row['Store ID']
                reorder_qty = row['Reorder Quantity']

                # Find matching product+store combination
                mask = (inventory_df['Product ID'] == product_id) & \
                       (inventory_df['Store ID'] == store_id)
                       
                if mask.any():
                    # Update inventory
                    inventory_df.loc[mask, 'Stock Levels'] += reorder_qty
                    
                    # Track processed reorder
                    updated_row = row.copy()
                    updated_row['Status'] = 'Auto-Reordered'
                    updated_row['ProcessedAt'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    updated_rows.append(updated_row)
                    processed_count += 1
                    self.logger.info(
                        f"Processed reorder for Product {product_id} at Store {store_id}, "
                        f"Qty: {reorder_qty}"
                    )
                else:
                    self.logger.warning(
                        f"No matching Product {product_id} at Store {store_id} in inventory"
                    )

            # Save updated data
            os.makedirs("output", exist_ok=True)
            inventory_df.to_csv("output/inventory_report.csv", index=False)
            
            result_df = pd.DataFrame(updated_rows)
            result_df.to_csv(self.output_path, index=False)
            
            self.logger.info(f"Completed {processed_count} auto-reorders")
            return result_df

        except Exception as e:
            self.logger.error(f"Failed to process reorders: {str(e)}")
            return None
