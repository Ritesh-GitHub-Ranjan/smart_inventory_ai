# 🧠 Smart Inventory Monitoring System

An AI-powered, multi-agent inventory optimization platform built for retail businesses. This system uses LLMs, demand forecasting, and autonomous agents to monitor stock, optimize pricing, automate reorders, and advise business decisions—all from a single intelligent dashboard.

---

## 📌 Features

| Feature                      | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **Demand Forecasting**      | Predicts future demand using historical data and external factors           |
| **Inventory Monitoring**     | Flags stockouts, expiry risks, and low inventory levels                    |
| **Reorder Optimization**     | Suggests optimal reorder quantities using safety stock and demand trends   |
| **Pricing Analysis**         | Detects underpriced/overpriced items based on elasticity and competitors   |
| **Sales Impact Estimation**  | Quantifies effect of pricing on sales volume                              |
| **Advisor Chat (LLM)**       | Answers inventory-related questions using on-prem LLMs (LLaMA, Mistral, etc.) |
| **Audit Logs**               | Tracks anomalies in stock levels and fulfillment processes                 |
| **Stock Redistribution**     | Suggests moving stock between stores to balance demand                     |
| **Streamlit Dashboard**      | Unified UI for control, insights, and decision-making                      |

---

## 📁 Project Structure

```
smart_inventory/
├── agents/
│   ├── advisor_agent.py
│   ├── audit_agent.py
│   ├── demand_agent.py
│   ├── forecasting_agent.py
│   ├── inventory_monitor.py
│   ├── pricing_agent.py
│   ├── reorder_agent.py
│   ├── sales_impact_agent.py
│   ├── auto_reorder.py
│   └── stock_redistribution_agent.py
├── data/
│   ├── demand_forecasting.csv
│   ├── inventory_monitoring.csv
│   └── pricing_optimization.csv
├── ml_models/
│   └── model.pkl  # Trained forecasting model
├── output/
│   ├── forecasted_demand.csv
│   ├── reorder_suggestions.csv
│   ├── pricing_analysis.csv
│   ├── sales_impact_analysis.csv
│   ├── inventory_audit_logs.csv
│   └── stock_redistribution.csv
├── utils/
│   └── helper_functions.py
├── app.py              # Streamlit dashboard
├── main.py             # Executes agents in sequence
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-inventory.git
cd smart-inventory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

### 4. Run Agents in Script Mode (Optional)
```bash
python main.py
```

---

## 🛠️ Key Technologies

- **Python**
- **Streamlit** (Dashboard UI)
- **Pandas / NumPy** (Data processing)
- **scikit-learn / XGBoost** (ML models)
- **SQLite** (Optional DB support)
- **LangChain + Ollama** (Advisor Agent using local LLMs)
- **Matplotlib / Seaborn** (Visualizations)

---

## 📊 Datasets Used

| File                      | Description                                         |
|--------------------------|-----------------------------------------------------|
| `demand_forecasting.csv` | Historical sales, price, promotions, trends         |
| `inventory_monitoring.csv` | Stock levels, lead time, expiry, warehouse capacity |
| `pricing_optimization.csv` | Product prices, reviews, return rate, competitors  |

---

## 🧩 Agent Descriptions

| Agent                | Role                                                                 |
|----------------------|----------------------------------------------------------------------|
| `ForecastingAgent`   | Predicts future demand using ML models                               |
| `InventoryMonitor`   | Flags low stock, expiring items, fulfillment delays                  |
| `ReorderAgent`       | Calculates reorder quantities using forecasts and safety stock       |
| `PricingAgent`       | Finds pricing inefficiencies using elasticity and sales ratios       |
| `SalesImpactAgent`   | Measures effect of price changes on sales volume                     |
| `AdvisorAgent`       | Interactive chat using on-prem LLMs                                  |
| `AuditAgent`         | Logs anomalies and inconsistencies                                   |
| `StockRedistribution`| Suggests inter-store inventory shifts to meet demand                 |
| `AutoReorder`        | Automatically updates stock levels in the system                     |

---

## 📺 Dashboard Overview

| Section                | Description                                               |
|------------------------|-----------------------------------------------------------|
| **Forecasted Demand**  | Line plots + demand predictions per product/store         |
| **Reorder Suggestions**| Reorder quantity and urgency flags                        |
| **Inventory Monitoring**| Stock levels, expiry alerts, fulfillment lags             |
| **Pricing Optimization**| Sales volume vs pricing patterns                          |
| **Sales Impact**       | Revenue gain/loss due to price changes                    |
| **Advisor Chat**       | Query any product insight from LLM (offline)              |
| **Audit Logs**         | See anomaly reports and stock fluctuations                |
| **Stock Redistribution** | Recommendations for moving stock between stores         |

---

## 🧠 LLM Integration (Advisor)

- Uses [Ollama](https://ollama.com) to run lightweight LLMs locally (phi3, mistral, llama3)
- Advisor can:
  - Explain sales anomalies
  - Interpret reorder recommendations
  - Advise price changes
  - Suggest strategies based on historical data

---

## 💡 Sample Use Cases

- A retail chain wants to **avoid stockouts** during a seasonal promotion
- Inventory manager needs to **know what to restock and when**
- Business team asks: "What happens if I raise prices by 10% for low-rated products?"
- Pricing team wants to know **price elasticity trends** by location
- Store manager wants to **move excess stock** from Store A to Store B

---

## 🧪 Testing (Optional)

Test each agent individually:
```bash
python -m agents.reorder_agent
python -m agents.inventory_monitor
# etc.
```

---

## 📣 Contribution

Pull requests welcome! For major changes, open an issue first to discuss what you’d like to change.

---

## 📄 License

MIT License

---