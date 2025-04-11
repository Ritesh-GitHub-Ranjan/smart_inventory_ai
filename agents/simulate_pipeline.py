import pandas as pd
from archive.forecasting_agent_v2 import ForecastingAgent
from agents.orders.reorder_agent import ReorderAgent
from agents.pricing_agent import PricingAgent
from agents.core.audit_agent import AuditAgent
from core_application.llama_integration import handle_user_query  # Optional: For simulation insights

def load_data():
    """
    Load the necessary data for the simulation.
    """
    forecast_df = pd.read_csv('data/demand_forecasting.csv')
    inventory_df = pd.read_csv('data/inventory_monitoring.csv')
    pricing_df = pd.read_csv('data/pricing_optimization.csv')
    return forecast_df, inventory_df, pricing_df

def run_forecasting_agent(forecasting_agent, forecast_df):
    """
    Run the forecasting agent to generate demand predictions.
    """
    print("Running forecasting agent...")
    forecast_results = forecasting_agent.predict_demand(forecast_df)
    print("Forecasting complete!")
    return forecast_results

def run_reorder_agent(reorder_agent, inventory_df, forecast_results):
    """
    Run the reorder agent to adjust inventory levels based on forecast.
    """
    print("Running reorder agent...")
    reorder_results = reorder_agent.adjust_inventory(inventory_df, forecast_results)
    print("Reordering complete!")
    return reorder_results

def run_pricing_agent(pricing_agent, pricing_df, forecast_results):
    """
    Run the pricing agent to adjust pricing based on forecast.
    """
    print("Running pricing agent...")
    pricing_results = pricing_agent.adjust_prices(pricing_df, forecast_results)
    print("Pricing update complete!")
    return pricing_results

def run_audit_agent(audit_agent, forecast_df, inventory_df, pricing_df):
    """
    Run the audit agent to perform system audit on the results.
    """
    print("Running audit agent...")
    audit_report = audit_agent.perform_audit(forecast_df, inventory_df, pricing_df)
    print("Audit complete!")
    return audit_report

def simulate_pipeline():
    """
    Simulate the entire pipeline: Forecast -> Reorder -> Price Update -> Audit.
    """
    # Load data
    forecast_df, inventory_df, pricing_df = load_data()

    # Initialize agents
    forecasting_agent = ForecastingAgent()
    reorder_agent = ReorderAgent()
    pricing_agent = PricingAgent()
    audit_agent = AuditAgent()

    # Run each agent in sequence
    forecast_results = run_forecasting_agent(forecasting_agent, forecast_df)
    reorder_results = run_reorder_agent(reorder_agent, inventory_df, forecast_results)
    pricing_results = run_pricing_agent(pricing_agent, pricing_df, forecast_results)
    audit_report = run_audit_agent(audit_agent, forecast_df, inventory_df, pricing_df)

    # Optional: Print the results for review
    print("\nFinal Audit Report:")
    print(audit_report)

    return {
        "forecast_results": forecast_results,
        "reorder_results": reorder_results,
        "pricing_results": pricing_results,
        "audit_report": audit_report
    }

if __name__ == "__main__":
    # Simulate the pipeline
    simulation_results = simulate_pipeline()

    # Optionally, save the results to CSV
    pd.DataFrame(simulation_results["forecast_results"]).to_csv('output/forecast_results.csv', index=False)
    pd.DataFrame(simulation_results["reorder_results"]).to_csv('output/reorder_results.csv', index=False)
    pd.DataFrame(simulation_results["pricing_results"]).to_csv('output/pricing_results.csv', index=False)
    pd.DataFrame(simulation_results["audit_report"]).to_csv('output/audit_report.csv', index=False)
    print("Simulation complete. Results saved to output directory.")
