# Smart Inventory AI - Code Structure Documentation

## Core Components

### 1. Main Application (app.py)
- Streamlit-based dashboard
- Entry point for the system
- Manages agent orchestration
- Handles user interactions

### 2. Agents Package
- **InventoryAgent**: Core monitoring logic
- **ReorderAgent**: Automated stock replenishment
- **PricingAgent**: Dynamic pricing strategies
- **InventoryMonitor**: Visualization and reporting
- **AdvisorAgent**: LLM-powered recommendations
- **AuditAgent**: Data validation and integrity
- **SalesImpactAgent**: Performance measurement

### 3. ML Models Package
- **Training**: demand_forecasting_model.py
- **Inference**: demand_forecasting_predict.py

### 4. Utils Package
- **stock_operations.py**: Core inventory operations
- **ollama_manager.py**: LLM integration layer

## Data Flow
1. Raw inventory data loaded from CSV/SQLite
2. Processed through agent pipeline
3. Results stored in output/ directory
4. Visualizations generated for dashboard

## Execution Sequence
1. Data loading and validation
2. Demand forecasting
3. Inventory analysis
4. Reorder calculations
5. Pricing optimization
6. Results aggregation
7. Dashboard updates
