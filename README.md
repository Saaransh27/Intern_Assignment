## 📈 Marketing Campaign Optimization Agent

This project provides a modular Python solution for **optimizing marketing budget allocation** to maximize conversions. It analyzes historical campaign data (`marketing_campaign_dataset.csv`) and uses a constrained optimization model (`scipy`) to recommend the most efficient spend across different channels.

### 📁 Project Structure

| File Name | Role | Description | 
| ----- | ----- | ----- | 
| **`marketing_campaign_dataset.csv`** | **Data Source** | Raw historical metrics (Spend, Impressions, Conversions). | 
| **`data_processor.py`** | **ETL** | Cleans data and engineers features for the model. | 
| **`optimization_core.py`** | **Core Logic** | Runs the mathematical optimization to calculate the optimal budget split. | 
| **`main_agent.py`** | **Entry Point** | Executes the full workflow and generates the strategic report. | 

### 🚀 Getting Started

#### Prerequisites

pandas
numpy
scipy 
scikit-learn 
matplotlib 
seaborn
