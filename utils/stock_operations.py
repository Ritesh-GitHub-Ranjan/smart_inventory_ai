def calculate_reorder_point(lead_time: int, daily_demand: int, safety_stock: int) -> int:
    """
    Calculate the reorder point for inventory management.
    
    Args:
        lead_time: Days between placing and receiving an order
        daily_demand: Average units consumed per day 
        safety_stock: Buffer stock to prevent stockouts
        
    Returns:
        The reorder point quantity (lead_time * daily_demand) + safety_stock
        
    Example:
        >>> calculate_reorder_point(7, 10, 20)
        90
    """
    if daily_demand < 0:
        daily_demand = 0
    if lead_time < 0:
        lead_time = 0
    if safety_stock < 0:
        safety_stock = 0
        
    return (lead_time * daily_demand) + safety_stock
