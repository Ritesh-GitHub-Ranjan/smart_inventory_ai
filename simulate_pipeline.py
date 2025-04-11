import pandas as pd
import pickle
from archive.forecasting_agent_v2 import ForecastingAgent
from agents.orders.reorder_agent import ReorderAgent
from agents.forecasting.pricing_agent import PricingAgent
from agents.core.audit_agent import AuditAgent
from agents.interfaces.advisor_agent import AdvisorAgent

# STEP 1: Load sample data
product_ids = [f"P00{i}" for i in range(1, 11)]
demand_df = pd.read_csv("data/demand_forecasting.csv")
inventory_df = pd.read_csv("data/inventory_monitoring.csv")

demand_sample = demand_df[demand_df["Product ID"].isin(product_ids)]
inventory_sample = inventory_df[inventory_df["Product ID"].isin(product_ids)]

# STEP 2: Forecast Demand
print("\n📈 Running Forecasting Agent...")
forecasting_agent = ForecastingAgent()
prediction_output = forecasting_agent.predict_demand(input_data=demand_sample)

if prediction_output["status"] == "success":
    forecast_cols = prediction_output["predictions"]
    forecast_df = pd.DataFrame(forecast_cols)
    forecasted_demand = pd.concat([demand_sample.reset_index(drop=True), forecast_df], axis=1)
    forecasted_demand.to_csv("output/predicted_demand.csv", index=False)
else:
    print("❌ Forecasting failed:", prediction_output["message"])
    exit(1)

# STEP 3: Reorder Products
print("\n📦 Running Reorder Agent...")
reorder_agent = ReorderAgent()
reorder_suggestions = reorder_agent.run(
    demand_forecast=forecasted_demand,
    inventory_data=inventory_sample
)
reorder_suggestions.to_csv("output/reorder_suggestions.csv", index=False)

# STEP 4: Optimize Pricing
print("\n💸 Running Pricing Agent...")
pricing_agent = PricingAgent()
pricing_result = pricing_agent.run(demand_forecast=forecasted_demand)
pricing_result.to_csv("output/pricing_optimization_results.csv", index=False)

# STEP 5: Audit System
print("\n🕵️ Running Audit Agent...")
audit_agent = AuditAgent()
audit_result = audit_agent.run()
audit_result.to_csv("output/audit_report.csv", index=False)

# STEP 6: Advisor Summary
print("\n🧠 Advisor Agent Summary:")
advisor = AdvisorAgent()
print(advisor.answer_query("Show me top 5 understocked products"))
