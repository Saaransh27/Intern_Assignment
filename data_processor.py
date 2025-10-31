import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# NOTE: optimization_core is accessed for global parameters

# =========================================================================
# --- Agent Configuration Parameters ---
# =========================================================================
# The FINAL Budget Fix (Set to $1 Million for meaningful results)
CSV_PATH = "marketing_campaign_dataset.csv" 
TOTAL_BUDGET = 1000000.0  # Constraint: Total budget for the new period (USD)

# Global variables (shared across modules after loading/training)
GLOBAL_CHANNEL_NAMES = []
GLOBAL_TARGET_NAME = 'Revenue'
GLOBAL_COEF_DICT = {}
GLOBAL_INTERCEPT = 0
MODEL_SCORE = 0

# =========================================================================
# HELPER FUNCTION: DATA PREPROCESSING
# =========================================================================
def load_and_prepare_data(csv_path, target_name_base='Revenue'):
    try:
        data = pd.read_csv(csv_path)
        print(f"✅ Data successfully loaded from {csv_path}. Shape: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("❌ Fatal Error: Could not load the required CSV file.")
        sys.exit(1)

    # 1. IDENTIFY AND CLEAN REQUIRED COLUMNS
    required_cols = ['Acquisition_Cost', 'ROI', 'Channel_Used', 'Date']
    if not all(col in data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in data.columns]
        print(f"❌ Fatal Error: Missing required columns in CSV: {missing_cols}")
        sys.exit(1)

    cost_col = 'Acquisition_Cost'
    roi_col = 'ROI'
    channel_col = 'Channel_Used'
    date_col = 'Date'

    data['Cost'] = data[cost_col].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)
    data['ROI_Value'] = data[roi_col]
    data[target_name_base] = data['Cost'] * (1 + data['ROI_Value'].clip(lower=-1.0))

    data['Date'] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=['Date', channel_col, 'Cost', target_name_base])
    
    # 2. AGGREGATION AND PIVOTING
    df_agg = data.groupby(['Date', channel_col]).agg(
        Total_Spend=('Cost', 'sum'),
        Total_Revenue=(target_name_base, 'sum')
    ).reset_index()
    
    df_spend_pivot = df_agg.pivot_table(
        index='Date', columns=channel_col, values='Total_Spend', fill_value=0
    )
    df_revenue = df_agg.groupby('Date')['Total_Revenue'].sum()
    df_final = df_spend_pivot.merge(df_revenue, left_index=True, right_index=True)
    df_final.rename(columns={'Total_Revenue': target_name_base}, inplace=True)
    
    global GLOBAL_CHANNEL_NAMES, GLOBAL_TARGET_NAME
    GLOBAL_CHANNEL_NAMES = df_spend_pivot.columns.tolist()
    GLOBAL_TARGET_NAME = target_name_base 
    
    historical_spend_sum = df_final[GLOBAL_CHANNEL_NAMES].sum().sum()
    historical_sales_sum = df_final[target_name_base].sum()

    print(f"✅ Data successfully aggregated to {len(df_final)} time steps (daily/weekly).")
    
    return df_final[GLOBAL_CHANNEL_NAMES], df_final[target_name_base], df_final, historical_spend_sum, historical_sales_sum

# =========================================================================
# HELPER FUNCTIONS: VISUALIZATION
# =========================================================================

def plot_historical_trends(X_data, y_data):
    df = X_data.copy()
    df['Total Spend'] = df.sum(axis=1)
    df[GLOBAL_TARGET_NAME] = y_data
    df['Date'] = df.index
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Date', y='Total Spend', data=df, label='Total Marketing Spend', linewidth=2)
    sns.lineplot(x='Date', y=GLOBAL_TARGET_NAME, data=df, label='Total Revenue (Sales)', linewidth=2)
    
    plt.title('1. Historical Spend vs. Revenue Trend', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Amount (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_budget_comparison_pie_charts(optimal_spend, even_split_spend):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.suptitle('3. Budget Allocation Comparison: Initial vs. Optimized', fontsize=16)

    def plot_pie(ax, data, title):
        ax.pie(
            data, labels=GLOBAL_CHANNEL_NAMES, autopct='%1.1f%%', startangle=90, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}, textprops={'fontsize': 10}
        )
        ax.set_title(title, fontsize=14)
        ax.axis('equal') 

    plot_pie(axes[0], even_split_spend, 'Initial Allocation Concept (Even Split)')
    plot_pie(axes[1], optimal_spend, "Agent's Optimal Budget Allocation")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_marginal_roi_bar_chart(recommendation_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='Channel', 
        y='MROI_Effectiveness_Value', 
        data=recommendation_df.sort_values(by='MROI_Effectiveness_Value', ascending=False),
        palette='viridis'
    )
    plt.title('2. Channel Benchmarks: Marginal ROI Effectiveness', fontsize=16)
    plt.xlabel('Marketing Channel', fontsize=12)
    plt.ylabel('Marginal ROI Effectiveness Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def get_final_summary_table(optimal_revenue, even_split_revenue, total_optimal_spend, total_budget):
    """Creates a comparison table focusing on financial results."""
    
    sales_gain = optimal_revenue - even_split_revenue
    
    summary_data = {
        'Metric': ['Projected Incremental Revenue', 'Total Budget Invested', 'Incremental ROI', 'Optimization Value Add'],
        'Optimized Allocation': [
            f"${optimal_revenue:,.2f}", 
            f"${total_optimal_spend:,.2f}",
            f"{(optimal_revenue / total_optimal_spend):.2f}",
            ""
        ],
        'Even Split Allocation': [
            f"${even_split_revenue:,.2f}", 
            f"${total_budget:,.2f}",
            f"{(even_split_revenue / total_budget):.2f}",
            f"+${sales_gain:,.2f}"
        ]
    }
    return pd.DataFrame(summary_data).set_index('Metric')

# This is the single plotting function for the final step, replacing the old comparison chart.
def plot_gain_bar_chart(optimal_revenue, even_split_revenue):
    """Plots the sales comparison focusing ONLY on the Optimization Value Add (Gain)."""
    
    gain = optimal_revenue - even_split_revenue
    
    sales_data = pd.DataFrame({
        'Metric': ['Optimization Value Add'],
        'Revenue Added': [gain]
    })
    
    plt.figure(figsize=(6, 6))
    sns.barplot(
        x='Metric', 
        y='Revenue Added', 
        data=sales_data, 
        palette=['#4CAF50']
    )
    
    # Annotation for Sales Gain
    plt.annotate(
        f"Gain: +${gain:,.2f}", 
        xy=(0, gain * 1.05),  # Positioned slightly above the bar
        ha='center', 
        fontsize=14, 
        color='black',
        fontweight='bold'
    )
    
    plt.title('4. Optimization Value Add (Projected Gain)', fontsize=16)
    plt.xlabel('', fontsize=12)
    plt.ylabel('Incremental Revenue (USD)', fontsize=12)
    plt.yticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()