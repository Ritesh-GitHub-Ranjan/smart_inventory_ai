import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch
from agents.core.audit_agent import AuditAgent
import logging

@pytest.fixture
def sample_inventory_data():
    """Sample inventory data for testing"""
    return pd.DataFrame({
        "Product ID": ["P1001", "P1002", "P1003"],
        "Store ID": ["S001", "S001", "S002"],
        "Stock Levels": [50, 10, 75],
        "Reorder Point": [30, 20, 50]
    })

@pytest.fixture
def sample_forecast_data():
    """Sample forecast data for testing"""
    return pd.DataFrame({
        "Product ID": ["P1001", "P1002", "P1003"],
        "Store ID": ["S001", "S001", "S002"],
        "Predicted Sales Quantity": [60, 25, 50]
    })

@pytest.fixture
def temp_output_dir():
    """Create and cleanup temp output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_initialization(temp_output_dir):
    """Test AuditAgent initialization"""
    # Test with custom paths
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    forecast_path = os.path.join(temp_output_dir, "forecast.csv")
    agent = AuditAgent(
        inventory_path=inventory_path,
        forecast_path=forecast_path
    )
    assert agent.inventory_path == inventory_path
    assert agent.forecast_path == forecast_path
    
    # Verify logger setup
    assert isinstance(agent.logger, logging.Logger)
    assert agent.logger.name == "AuditAgent"

def test_run_analysis(sample_inventory_data, sample_forecast_data, temp_output_dir):
    """Test full audit analysis workflow"""
    # Setup test files
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    forecast_path = os.path.join(temp_output_dir, "forecast.csv")
    sample_inventory_data.to_csv(inventory_path, index=False)
    sample_forecast_data.to_csv(forecast_path, index=False)
    
    # Run agent
    with patch("pandas.read_csv", side_effect=[sample_inventory_data, sample_forecast_data]):
        agent = AuditAgent(
            inventory_path=inventory_path,
            forecast_path=forecast_path,
        )
        agent.run()
        
        # Verify output
        output_path = os.path.join(temp_output_dir, "output", "inventory_audit.csv")
        assert os.path.exists(output_path)
        
        # Verify mismatches detected (P1001 and P1002 should be flagged)
        audit_df = pd.read_csv(output_path)
        assert len(audit_df) == 2
        assert "P1001" in audit_df["Product ID"].values
        assert "P1002" in audit_df["Product ID"].values

def test_gap_calculation(sample_inventory_data, sample_forecast_data):
    """Test stock gap calculations"""
    agent = AuditAgent()
    
    # Merge test data
    merged = pd.merge(
        sample_inventory_data,
        sample_forecast_data,
        on=["Product ID", "Store ID"]
    )
    
    # Test gap calculation
    merged["Stock Gap"] = (merged["Stock Levels"] - merged["Predicted Sales Quantity"]).abs()
    assert merged.loc[0, "Stock Gap"] == 10  # |50-60|
    assert merged.loc[1, "Stock Gap"] == 15  # |10-25|

def test_mismatch_detection(sample_inventory_data, sample_forecast_data):
    """Test mismatch detection logic"""
    agent = AuditAgent(threshold=15)  # Set custom threshold
    
    # Merge test data
    merged = pd.merge(
        sample_inventory_data,
        sample_forecast_data,
        on=["Product ID", "Store ID"]
    )
    
    # Calculate gaps
    merged["Stock Gap"] = (merged["Stock Levels"] - merged["Predicted Sales Quantity"]).abs()
    
    # Test detection
    mismatches = merged[merged["Stock Gap"] > agent.threshold]
    assert len(mismatches) == 1  # Only P1002 (gap=15) should be flagged with threshold=15
    assert mismatches.iloc[0]["Product ID"] == "P1002"

def test_error_handling(temp_output_dir):
    """Test error cases"""
    # Test missing files
    with patch("os.path.exists", return_value=False):
        agent = AuditAgent()
        with pytest.raises(Exception):
            agent.run()
    
    # Test invalid data format
    bad_data_path = os.path.join(temp_output_dir, "bad_data.csv")
    pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(bad_data_path, index=False)
    
    with patch("pandas.read_csv", return_value=pd.read_csv(bad_data_path)):
        agent = AuditAgent(
            inventory_path=bad_data_path,
            forecast_path=bad_data_path
        )
        with pytest.raises(Exception):
            agent.run()

def test_output_directory_creation(temp_output_dir):
    """Test output directory is created if missing"""
    # Setup test files
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    forecast_path = os.path.join(temp_output_dir, "forecast.csv")
    pd.DataFrame().to_csv(inventory_path, index=False)
    pd.DataFrame().to_csv(forecast_path, index=False)
    
    # Remove output directory if exists
    output_dir = os.path.join(temp_output_dir, "output")
    if os.path.exists(output_dir):
        os.rmdir(output_dir)
    
    with patch("pandas.read_csv", return_value=pd.DataFrame()):
        agent = AuditAgent(
            inventory_path=inventory_path,
            forecast_path=forecast_path
        )
        agent.run()
        
        # Verify directory was created
        assert os.path.exists(output_dir)
        assert os.path.isdir(output_dir)
