import pandas as pd
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List, Union


class ReorderAgent:
    def __init__(
        self,
        inventory_path: str = None,
        forecast_df: pd.DataFrame = None,
        output_path: str = "output/reorder_suggestions.csv"
    ):
        """Initialize ReorderAgent with inventory and demand data sources.
        
        Args:
            inventory_path: Path to inventory CSV file
            forecast_df: Pre-loaded demand forecast DataFrame
            output_path: Path to save reorder suggestions
        """
        if not inventory_path:
            raise ValueError("inventory_path is required")

        try:
            self.inventory_df = pd.read_csv(inventory_path)
        except Exception as e:
            raise ValueError(f"Failed to load inventory data: {str(e)}")

        # If forecast_df is passed directly, use it
        if forecast_df is not None:
            self.demand_df = forecast_df
        else:
            raise ValueError("forecast_df must be provided.")

        self.output_path = output_path

        # Configure logging
        self.logger = logging.getLogger("ReorderAgent")
        self.logger.setLevel(logging.INFO)

    def calculate_reorder_quantity(self, row: Dict) -> int:
        """Calculate optimal reorder quantity for a product.
        
        Args:
            row: Dictionary containing product inventory and demand data
            
        Returns:
            Calculated reorder quantity (rounded to integer)
        """
        current_stock = row["Stock Levels"]
        predicted_demand = row.get("Predicted Sales Quantity") or row.get("Predicted Demand")

        if pd.isna(predicted_demand):
            self.logger.warning(f"Missing demand for Product {row['Product ID']}")
            return 0

        safety_stock = predicted_demand * 0.2
        reorder_qty = predicted_demand + safety_stock - current_stock
        reorder_qty = max(0, reorder_qty)

        self.logger.info(
            f"Product {row['Product ID']} | Stock={current_stock}, Demand={predicted_demand}, "
            f"Safety Stock={int(safety_stock)}, Reorder Qty={int(reorder_qty)}"
        )

        return int(reorder_qty)

    def analyze_reorder_needs(self) -> pd.DataFrame:
        """Analyze inventory and identify products needing reorder.
        
        Returns:
            DataFrame of reorder suggestions
        """
        self.logger.info("Analyzing reorder needs...")
        
        # Log initial data info
        self.logger.info(f"Forecast DF shape: {self.demand_df.shape}")
        self.logger.info(f"Inventory DF shape: {self.inventory_df.shape}")
        self.logger.info(f"Forecast DF columns: {self.demand_df.columns.tolist()}")
        self.logger.info(f"Inventory DF columns: {self.inventory_df.columns.tolist()}")

        # Validate Store IDs
        if 'Store ID' not in self.inventory_df.columns:
            raise ValueError("Inventory data missing Store ID column")
            
        if self.inventory_df['Store ID'].isna().any():
            raise ValueError("Inventory data contains empty Store IDs")
            
        # Ensure consistent data types for merging
        self.demand_df['Store ID'] = self.demand_df['Store ID'].astype(str)
        self.inventory_df['Store ID'] = self.inventory_df['Store ID'].astype(str)
        
        # Log Store ID info
        unique_store_ids = self.inventory_df['Store ID'].unique()
        self.logger.info(f"Found {len(unique_store_ids)} unique Store IDs: {unique_store_ids}")
        
        # Ensure required columns exist
        required_cols = ["Product ID", "Store ID", "Stock Levels", "Reorder Point"]
        if not all(col in self.inventory_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.inventory_df.columns]
            raise ValueError(f"Inventory data missing required columns: {missing}")

        # Handle Store ID in forecast data
        if "Store ID" not in self.demand_df.columns:
            self.logger.warning("Forecast data missing Store ID - using first inventory Store ID")
            self.demand_df["Store ID"] = self.inventory_df["Store ID"].iloc[0]
        elif self.demand_df['Store ID'].isna().any():
            raise ValueError("Forecast data contains empty Store IDs")

        # Merge with enhanced error handling
        try:
            # First try exact Store ID matches
            merged = pd.merge(
                self.demand_df,
                self.inventory_df,
                on=["Product ID", "Store ID"],
                how="inner"
            )
            
            if merged.empty:
                # Fallback - try matching just on Product ID
                self.logger.warning("No matches on Store ID - falling back to Product ID only")
                merged = pd.merge(
                    self.demand_df,
                    self.inventory_df,
                    on="Product ID",
                    how="inner"
                )
                
                if merged.empty:
                    self.logger.error("No matching Product IDs found between datasets")
                    return pd.DataFrame()
                else:
                    self.logger.warning(f"Found {len(merged)} matches on Product ID only")
        except Exception as e:
            self.logger.error(f"Merge failed: {e}")
            raise

        reorder_items = []

        for _, row in merged.iterrows():
            predicted_demand = row.get('Predicted Sales Quantity') or row.get('Predicted Demand')
            stock = row.get('Stock Levels')
            reorder_point = row.get('Reorder Point')

            if pd.isna(predicted_demand) or pd.isna(stock) or pd.isna(reorder_point):
                self.logger.warning(f"Skipping row due to missing values: {row.to_dict()}")
                continue

            if stock < reorder_point or stock < predicted_demand:
                reorder_qty = self.calculate_reorder_quantity(row)

                reorder_items.append({
                    "Product ID": row["Product ID"],
                    "Store ID": row["Store ID"],
                    "Stock Levels": stock,
                    "Predicted Demand": predicted_demand,
                    "Lead Time": row['Supplier Lead Time (days)'],
                    "Safety Stock": int(predicted_demand * 0.2),
                    "Suggested Reorder Qty": reorder_qty
                })

        reorder_df = pd.DataFrame(reorder_items).sort_values(by="Stock Levels")
        os.makedirs("output", exist_ok=True)
        reorder_df.to_csv(self.output_path, index=False)
        
        self.logger.info(f"📦 Generated {len(reorder_df)} reorder suggestions")
        print("📦 Reorder suggestions saved to", self.output_path)
        
        return reorder_df

    def trigger_auto_reorders(self, reorder_df: pd.DataFrame) -> None:
        """Trigger automatic reorders for low-stock items.
        
        Args:
            reorder_df: DataFrame of reorder suggestions
        """
        self.logger.info("Triggering auto-reorders...")
        auto_reorders = reorder_df.rename(columns={"Reorder Quantity": "Suggested Reorder Qty"}).copy()
        auto_reorders["Triggered At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        auto_reorders["Triggered By"] = "Auto-System"

        auto_path = "output/auto_reorders.csv"
        if os.path.exists(auto_path):
            existing = pd.read_csv(auto_path)
            auto_reorders = pd.concat([existing, auto_reorders], ignore_index=True)

        auto_reorders.to_csv(auto_path, index=False)
        print("Auto-reorders logged to", auto_path)
        self.logger.info("Auto-reorders logged successfully.")

    def run(self) -> pd.DataFrame:
        """Execute full reorder analysis workflow.
        
        Returns:
            DataFrame of reorder suggestions
        """
        self.logger.info("Running ReorderAgent...")
        reorder_df = self.analyze_reorder_needs()
        if not reorder_df.empty:
            self.trigger_auto_reorders(reorder_df)
        self.logger.info("Finished ReorderAgent.")
        return reorder_df
