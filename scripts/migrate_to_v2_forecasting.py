"""
Migration script to upgrade from v1 to v2 demand forecasting system.
This script handles:
1. Training the new ensemble model
2. Verifying model performance
3. Updating dependent components
4. Creating backup of previous model
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from ml_models.demand_forecasting_v2 import DemandForecaster
from archive.forecasting_agent_v2 import ForecastingAgent
import logging
import shutil
import os
from datetime import datetime

# Configuration
MODEL_V1_PATH = "ml_models/model.pkl"
MODEL_V2_PATH = "ml_models/ensemble_model.pkl"
DATA_PATH = "data/demand_forecasting.csv"
BACKUP_DIR = "model_backups"

def setup_logging():
    """Configure logging for the migration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('migration.log'),
            logging.StreamHandler()
        ]
    )

def backup_previous_model():
    """Create backup of previous model"""
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        if os.path.exists(MODEL_V1_PATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(BACKUP_DIR, f"model_v1_{timestamp}.pkl")
            shutil.copy2(MODEL_V1_PATH, backup_path)
            logging.info(f"Created backup of v1 model at {backup_path}")
        else:
            logging.warning("No v1 model found to backup")
    except Exception as e:
        logging.error(f"Failed to create backup: {str(e)}")
        raise

def train_new_model():
    """Train and evaluate the new ensemble model"""
    try:
        logging.info("Loading training data...")
        df = pd.read_csv(DATA_PATH)
        
        # Add new features with default values if they don't exist
        if 'Promotion Budget' not in df.columns:
            df['Promotion Budget'] = 0
        if 'Promotion Days' not in df.columns:
            df['Promotion Days'] = 0
        if 'Weather Index' not in df.columns:
            df['Weather Index'] = 100  # Default neutral weather
        if 'Economic Index' not in df.columns:
            df['Economic Index'] = 100  # Default neutral economy
        if 'Seasonality Factors' not in df.columns:
            df['Seasonality Factors'] = 'None'  # Default no seasonality
        if 'Demand Trend' not in df.columns:
            df['Demand Trend'] = 'Stable'  # Default stable trend
            
        logging.info("Training new ensemble model...")
        forecaster = DemandForecaster()
        forecaster.train_ensemble(df, save_path=MODEL_V2_PATH)
        
        logging.info("Evaluating new model...")
        metrics = forecaster.evaluate(df)
        logging.info(f"Model evaluation metrics:\n{pd.Series(metrics).to_string()}")
        
        return True
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        return False

def verify_agent_integration():
    """Verify the new agent works with the model"""
    try:
        logging.info("Testing agent integration...")
        agent = ForecastingAgent()
        
        # Test with sample data
        sample_data = pd.read_csv(DATA_PATH).head(10)
        prediction = agent.predict_demand(sample_data)
        
        if prediction['status'] != 'success':
            raise RuntimeError("Agent prediction failed")
            
        logging.info("Agent integration verified successfully")
        return True
    except Exception as e:
        logging.error(f"Agent verification failed: {str(e)}")
        return False

def update_dependencies():
    """Update dependent components to use new model"""
    try:
        logging.info("Updating dependent components...")
        
        # Example: Update forecasting agent reference in main system
        # This would depend on your actual system architecture
        # Here we just log the need to update references
        logging.warning("Remember to update system components to use ForecastingAgentV2")
        
        return True
    except Exception as e:
        logging.error(f"Dependency update failed: {str(e)}")
        return False

def main():
    """Run the migration process"""
    setup_logging()
    logging.info("Starting migration to v2 forecasting system")
    
    try:
        # Step 1: Backup current model
        backup_previous_model()
        
        # Step 2: Train new model
        if not train_new_model():
            raise RuntimeError("Model training failed")
            
        # Step 3: Verify integration
        if not verify_agent_integration():
            raise RuntimeError("Integration verification failed")
            
        # Step 4: Update dependencies
        if not update_dependencies():
            raise RuntimeError("Dependency update failed")
            
        logging.info("Migration completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Migration failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
