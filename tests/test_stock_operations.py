import pytest
from utils.stock_operations import calculate_reorder_point
from tests.conftest import db_session

def test_calculate_reorder_point(db_session):
    """Test reorder point calculation"""
    # Test with typical values
    result = calculate_reorder_point(
        lead_time=7,
        daily_demand=10,
        safety_stock=20
    )
    assert result == 90  # (7 * 10) + 20

def test_calculate_reorder_point_edge_cases(db_session):
    """Test edge cases in reorder calculation"""
    # Test with zero values
    result = calculate_reorder_point(
        lead_time=0,
        daily_demand=10,
        safety_stock=20
    )
    assert result == 20
    
    # Test with negative values (should handle gracefully)
    result = calculate_reorder_point(
        lead_time=7,
        daily_demand=-5,
        safety_stock=20
    )
