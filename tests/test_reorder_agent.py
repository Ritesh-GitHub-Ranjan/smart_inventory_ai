import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch
from agents.orders.reorder_agent import ReorderAgent
import logging

@pytest.fixture
def sample_inventory_data():
    """Sample inventory data for testing"""
    return pd.DataFrame({
        "Product ID": ["P1001", "P1002", "P1003"],
        "Store ID": ["S001", "S001", "S002"],
        "Stock Levels": [50, 10, 75],
        "Reorder Point": [30, 20, 50],
        "Supplier Lead Time (days)": [7, 5, 3]
    })

@pytest.fixture
def sample_forecast_data():
    """Sample demand forecast data for testing"""
    return pd.DataFrame({
        "Product ID": ["P1001", "P1002", "P1003"],
        "Store ID": ["S001", "S001", "S002"],
        "Predicted Demand": [60, 25, 50]
    })

@pytest.fixture
def temp_output_dir():
    """Create and cleanup temp output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_agent_initialization(sample_inventory_data, sample_forecast_data, temp_output_dir):
    """Test ReorderAgent initialization"""
    # Test valid initialization
    output_path = os.path.join(temp_output_dir, "reorders.csv")
    agent = ReorderAgent(
        inventory_path="test_inventory.csv",
        forecast_df=sample_forecast_data,
        output_path=output_path
    )
    assert agent is not None
    assert agent.output_path == output_path

    # Test missing inventory path
    with pytest.raises(ValueError):
        ReorderAgent(forecast_df=sample_forecast_data)

    # Test missing forecast data
    with pytest.raises(ValueError):
        ReorderAgent(inventory_path="test_inventory.csv")

def test_calculate_reorder_quantity(sample_inventory_data, sample_forecast_data):
    """Test reorder quantity calculations"""
    agent = ReorderAgent(
        inventory_path="test_inventory.csv",
        forecast_df=sample_forecast_data
    )
    
    # Test normal case
    row = {
        "Product ID": "P1001",
        "Stock Levels": 50,
        "Predicted Demand": 60,
        "Reorder Point": 30
    }
    assert agent.calculate_reorder_quantity(row) == 22  # (60 + 12) - 50

    # Test zero demand
    row["Predicted Demand"] = 0
    assert agent.calculate_reorder_quantity(row) == 0

    # Test sufficient stock
    row["Stock Levels"] = 100
    row["Predicted Demand"] = 50
    assert agent.calculate_reorder_quantity(row) == 0

def test_analyze_reorder_needs(sample_inventory_data, sample_forecast_data, temp_output_dir):
    """Test full reorder analysis workflow"""
    output_path = os.path.join(temp_output_dir, "reorders.csv")
    
    # Mock file reading
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        agent = ReorderAgent(
            inventory_path="test_inventory.csv",
            forecast_df=sample_forecast_data,
            output_path=output_path
        )
        
        result = agent.analyze_reorder_needs()
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # P1002 and P1001 should need reorder
        assert "P1002" in result["Product ID"].values
        assert os.path.exists(output_path)

def test_trigger_auto_reorders(sample_inventory_data, sample_forecast_data, temp_output_dir):
    """Test auto-reorder triggering"""
    output_path = os.path.join(temp_output_dir, "reorders.csv")
    auto_path = os.path.join(temp_output_dir, "auto_reorders.csv")
    
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        agent = ReorderAgent(
            inventory_path="test_inventory.csv",
            forecast_df=sample_forecast_data,
            output_path=output_path
        )
        
        # First run - should create file
        reorder_df = agent.analyze_reorder_needs()
        agent.trigger_auto_reorders(reorder_df)
        assert os.path.exists(auto_path)
        
        # Second run - should append
        initial_count = len(pd.read_csv(auto_path))
        agent.trigger_auto_reorders(reorder_df)
        updated_count = len(pd.read_csv(auto_path))
        assert updated_count == initial_count * 2

def test_error_handling(sample_inventory_data, sample_forecast_data):
    """Test error cases and edge conditions"""
    # Test missing required columns
    bad_inventory = sample_inventory_data.drop(columns=["Stock Levels"])
    with patch("pandas.read_csv", return_value=bad_inventory):
        with pytest.raises(ValueError):
            agent = ReorderAgent(
                inventory_path="test_inventory.csv",
                forecast_df=sample_forecast_data
            )
            agent.analyze_reorder_needs()

    # Test empty merge result
    no_match_forecast = sample_forecast_data.copy()
    no_match_forecast["Product ID"] = "P9999"
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        agent = ReorderAgent(
            inventory_path="test_inventory.csv",
            forecast_df=no_match_forecast
        )
        result = agent.analyze_reorder_needs()
        assert result.empty

def test_full_run(sample_inventory_data, sample_forecast_data, temp_output_dir):
    """Test complete agent execution"""
    output_path = os.path.join(temp_output_dir, "reorders.csv")
    auto_path = os.path.join(temp_output_dir, "auto_reorders.csv")
    
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        agent = ReorderAgent(
            inventory_path="test_inventory.csv",
            forecast_df=sample_forecast_data,
            output_path=output_path
        )
        
        # Capture logs to verify output
        with patch.object(agent.logger, "info") as mock_logger:
            result = agent.run()
            
            # Verify logging occurred
            assert mock_logger.call_count >= 3
            assert "Running ReorderAgent" in mock_logger.call_args_list[0].args[0]
            assert "Finished ReorderAgent" in mock_logger.call_args_list[-1].args[0]
            
        # Verify outputs
        assert isinstance(result, pd.DataFrame)
        assert os.path.exists(output_path)
        assert os.path.exists(auto_path)
