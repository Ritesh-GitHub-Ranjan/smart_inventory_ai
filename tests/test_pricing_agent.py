import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch
from agents.forecasting.pricing_agent import PricingAgent
import logging

@pytest.fixture
def sample_pricing_data():
    """Sample pricing data for testing"""
    return pd.DataFrame({
        "Product ID": ["P1001", "P1002", "P1003", "P1004"],
        "Price": [100, 50, 200, 75],
        "Sales Volume": [50, 200, 100, 50],
        "Cost": [40, 20, 80, 30]
    })

@pytest.fixture
def temp_output_dir():
    """Create and cleanup temp output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_initialization(temp_output_dir):
    """Test PricingAgent initialization"""
    # Test with custom path
    data_path = os.path.join(temp_output_dir, "pricing.csv")
    agent = PricingAgent(data_path=data_path)
    assert agent.data_path == data_path
    
    # Verify logger setup
    assert isinstance(agent.logger, logging.Logger)
    assert agent.logger.name == "PricingAgent"

def test_run_analysis(sample_pricing_data, temp_output_dir):
    """Test full pricing analysis workflow"""
    # Setup test file
    data_path = os.path.join(temp_output_dir, "pricing.csv")
    sample_pricing_data.to_csv(data_path, index=False)
    
    # Run agent
    with patch("pandas.read_csv", return_value=sample_pricing_data):
        agent = PricingAgent(data_path=data_path)
        agent.run()
        
        # Verify outputs
        overpriced_path = os.path.join(temp_output_dir, "output", "overpriced_products.csv")
        underpriced_path = os.path.join(temp_output_dir, "output", "underpriced_products.csv")
        
        assert os.path.exists(overpriced_path)
        assert os.path.exists(underpriced_path)
        
        # Verify correct products were flagged
        overpriced = pd.read_csv(overpriced_path)
        underpriced = pd.read_csv(underpriced_path)
        
        assert len(overpriced) == 1  # P1003 (200/100=2.0)
        assert len(underpriced) == 1  # P1002 (50/200=0.25)
        assert "P1003" in overpriced["Product ID"].values
        assert "P1002" in underpriced["Product ID"].values

def test_price_to_sales_calculation(sample_pricing_data):
    """Test price-to-sales ratio calculations"""
    agent = PricingAgent()
    
    # Test ratio calculation
    sample_pricing_data = agent._calculate_ratios(sample_pricing_data)
    assert "Price-to-Sales Ratio" in sample_pricing_data.columns
    assert sample_pricing_data.loc[0, "Price-to-Sales Ratio"] == 2.0  # 100/50
    assert sample_pricing_data.loc[1, "Price-to-Sales Ratio"] == 0.25  # 50/200

def test_threshold_detection(sample_pricing_data):
    """Test overpriced/underpriced detection"""
    agent = PricingAgent()
    
    # Add ratio column
    sample_pricing_data["Price-to-Sales Ratio"] = (
        sample_pricing_data["Price"] / sample_pricing_data["Sales Volume"]
    )
    
    # Test detection
    overpriced, underpriced = agent._detect_price_issues(sample_pricing_data)
    
    assert len(overpriced) == 1
    assert len(underpriced) == 1
    assert overpriced.iloc[0]["Product ID"] == "P1003"
    assert underpriced.iloc[0]["Product ID"] == "P1002"

def test_error_handling(temp_output_dir):
    """Test error cases"""
    # Test invalid data path
    with pytest.raises(Exception):
        agent = PricingAgent(data_path="nonexistent.csv")
        agent.run()
    
    # Test invalid data format
    bad_data_path = os.path.join(temp_output_dir, "bad_pricing.csv")
    pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(bad_data_path, index=False)
    
    with patch("pandas.read_csv", return_value=pd.read_csv(bad_data_path)):
        agent = PricingAgent(data_path=bad_data_path)
        with pytest.raises(Exception):
            agent.run()

def test_output_directory_creation(sample_pricing_data, temp_output_dir):
    """Test output directory is created if missing"""
    data_path = os.path.join(temp_output_dir, "pricing.csv")
    sample_pricing_data.to_csv(data_path, index=False)
    
    # Remove output directory if exists
    output_dir = os.path.join(temp_output_dir, "output")
    if os.path.exists(output_dir):
        os.rmdir(output_dir)
    
    with patch("pandas.read_csv", return_value=sample_pricing_data):
        agent = PricingAgent(data_path=data_path)
        agent.run()
        
        # Verify directory was created
        assert os.path.exists(output_dir)
        assert os.path.isdir(output_dir)
