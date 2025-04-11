import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
from agents.core.inventory_monitor import InventoryMonitor
import logging
import matplotlib.pyplot as plt

@pytest.fixture
def sample_inventory_data():
    """Sample inventory data for testing"""
    return pd.DataFrame({
        "Product ID": ["P1001", "P1002", "P1003", "P1004"],
        "Store ID": ["S001", "S001", "S002", "S002"],
        "Stock Levels": [15, 50, 10, 30],
        "Reorder Point": [20, 30, 15, 25],
        "Expiry Date": ["2024-12-31", "2024-06-15", "2024-05-01", "2024-07-30"],
        "Warehouse Capacity": [15, 80, 5, 50]
    })

@pytest.fixture
def temp_output_dir():
    """Create and cleanup temp output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_reorder_agent():
    """Mock ReorderAgent to prevent actual execution"""
    with patch("agents.orders.reorder_agent.ReorderAgent") as mock:
        instance = mock.return_value
        instance.run.return_value = pd.DataFrame()
        yield mock

def test_initialization(temp_output_dir):
    """Test InventoryMonitor initialization"""
    # Test with custom path
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    monitor = InventoryMonitor(inventory_path=inventory_path)
    assert monitor.inventory_path == inventory_path
    
    # Verify logger setup
    assert isinstance(monitor.logger, logging.Logger)
    assert monitor.logger.name == "InventoryMonitor"

def test_run_analysis(sample_inventory_data, temp_output_dir, mock_reorder_agent):
    """Test full analysis workflow"""
    # Setup test files
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    sample_inventory_data.to_csv(inventory_path, index=False)
    
    forecast_path = os.path.join(temp_output_dir, "forecasted_demand.csv")
    pd.DataFrame({
        "Product ID": ["P1001", "P1002", "P1003"],
        "Predicted Demand": [25, 40, 20]
    }).to_csv(forecast_path, index=False)
    
    # Run monitor
    with patch("pandas.read_csv", side_effect=[sample_inventory_data, pd.read_csv(forecast_path)]):
        monitor = InventoryMonitor(inventory_path=inventory_path)
        monitor.run()
        
        # Verify outputs
        report_path = os.path.join(temp_output_dir, "output", "inventory_report.csv")
        assert os.path.exists(report_path)
        
        dashboard_path = os.path.join(temp_output_dir, "output", "inventory_dashboard.html")
        assert os.path.exists(dashboard_path)
        
        # Verify charts were created
        assert os.path.exists(os.path.join(temp_output_dir, "output", "stock_vs_reorder.png"))
        assert os.path.exists(os.path.join(temp_output_dir, "output", "expiry_pie.png"))
        assert os.path.exists(os.path.join(temp_output_dir, "output", "capacity_histogram.png"))
        
        # Verify ReorderAgent was called for stockout items
        mock_reorder_agent.assert_called_once()

def test_stockout_detection(sample_inventory_data, temp_output_dir):
    """Test stockout risk detection"""
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    sample_inventory_data.to_csv(inventory_path, index=False)
    
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        monitor = InventoryMonitor(inventory_path=inventory_path)
        
        # Mock ReorderAgent to verify it's called
        with patch("agents.core.inventory_monitor.ReorderAgent") as mock_agent:
            instance = mock_agent.return_value
            instance.run.return_value = pd.DataFrame()
            
            monitor.run()
            
            # Should call ReorderAgent for P1001 and P1003 (stock < reorder point)
            mock_agent.assert_called_once()
            assert instance.run.called

def test_expiry_detection(sample_inventory_data, temp_output_dir):
    """Test expiring product detection"""
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    sample_inventory_data.to_csv(inventory_path, index=False)
    
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        monitor = InventoryMonitor(inventory_path=inventory_path)
        monitor.run()
        
        # Verify report contains expiring items
        report_path = os.path.join(temp_output_dir, "output", "inventory_report.csv")
        with open(report_path) as f:
            report = f.read()
            assert "Expiring soon" in report

def test_chart_generation(sample_inventory_data, temp_output_dir):
    """Test visualization chart generation"""
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    sample_inventory_data.to_csv(inventory_path, index=False)
    
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        # Mock plt to verify chart generation
        with patch("matplotlib.pyplot") as mock_plt:
            mock_plt.figure.return_value = MagicMock()
            monitor = InventoryMonitor(inventory_path=inventory_path)
            monitor.run()
            
            # Verify chart generation calls
            assert mock_plt.figure.call_count >= 3
            assert mock_plt.bar.called or mock_plt.pie.called or mock_plt.hist.called

def test_error_handling(temp_output_dir):
    """Test error cases"""
    # Test invalid inventory path
    with pytest.raises(Exception):
        monitor = InventoryMonitor(inventory_path="nonexistent.csv")
        monitor.run()
    
    # Test invalid data format
    bad_data_path = os.path.join(temp_output_dir, "bad_inventory.csv")
    pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(bad_data_path, index=False)
    
    with patch("pandas.read_csv", return_value=pd.read_csv(bad_data_path)):
        monitor = InventoryMonitor(inventory_path=bad_data_path)
        with pytest.raises(Exception):
            monitor.run()

def test_dashboard_creation(sample_inventory_data, temp_output_dir):
    """Test HTML dashboard generation"""
    inventory_path = os.path.join(temp_output_dir, "inventory.csv")
    sample_inventory_data.to_csv(inventory_path, index=False)
    
    with patch("pandas.read_csv", return_value=sample_inventory_data):
        # Mock webbrowser to prevent actual opening
        with patch("webbrowser.open"):
            monitor = InventoryMonitor(inventory_path=inventory_path)
            monitor.run()
            
            # Verify dashboard content
            dashboard_path = os.path.join(temp_output_dir, "output", "inventory_dashboard.html")
            with open(dashboard_path) as f:
                content = f.read()
                assert "<title>Inventory Dashboard</title>" in content
                assert "Stock Levels vs Reorder Point" in content
