import pandas as pd
import numpy as np
import data_processor as dp 
import optimization_core as opt 

# =========================================================================
# MAIN REPORT DRIVER
# =========================================================================

def generate_report(X_data, df_final, optimal_spend, optimal_revenue, total_budget):
    
    total_optimal_spend = np.sum(optimal_spend)
    
    # 1. Historical Benchmarks
    historical_spend = df_final[dp.GLOBAL_CHANNEL_NAMES].sum().sum()
    historical_revenue = df_final[dp.GLOBAL_TARGET_NAME].sum()
    overall_hist_roi = (historical_revenue - historical_spend) / historical_spend if historical_spend > 0 else np.nan

    # 2. Even Split Comparison
    even_split_spend = np.array([total_budget / len(dp.GLOBAL_CHANNEL_NAMES)] * len(dp.GLOBAL_CHANNEL_NAMES))
    log_even_split = np.log1p(even_split_spend)
    
    # Calculate predicted revenue for the even split (must also ignore intercept)
    log_coefficients = np.array([dp.GLOBAL_COEF_DICT[f'Log_{c}'] for c in dp.GLOBAL_CHANNEL_NAMES])
    even_split_revenue = (log_coefficients * log_even_split).sum() 
    
    sales_gain = optimal_revenue - even_split_revenue
    
    # 3. MROI Benchmarks for Reporting
    MROI_BASE = 100.0
    mroi_data = []
    
    for channel in dp.GLOBAL_CHANNEL_NAMES:
        mroi_value = opt.calculate_marginal_roi(channel, MROI_BASE)
        mroi_data.append({'Channel': channel, 'MROI_Effectiveness_Value': mroi_value, 'MROI_Effectiveness': f"{mroi_value:.3f}"})

    recommendation_df = pd.DataFrame(mroi_data).merge(
        pd.DataFrame({
            'Channel': dp.GLOBAL_CHANNEL_NAMES,
            'Recommended_Budget ($)': optimal_spend.round(2),
            'Even_Split_Budget ($)': even_split_spend.round(2),
        }),
        on='Channel'
    )
    
    # 4. Generate Financial Summary Table
    summary_table = dp.get_final_summary_table(
        optimal_revenue, even_split_revenue, total_optimal_spend, total_budget
    )
    
    # =================================================================
    # VISUALIZATION CALLS (Step-by-Step Presentation)
    # =================================================================
    print("\n--- Generating Visualizations (4 Charts) ---")
    dp.plot_historical_trends(X_data, df_final[dp.GLOBAL_TARGET_NAME])
    dp.plot_marginal_roi_bar_chart(recommendation_df)
    dp.plot_budget_comparison_pie_charts(optimal_spend, even_split_spend)
    
    # FINAL VISUAL: Uses the new gain-focused plot
    dp.plot_gain_bar_chart(optimal_revenue, even_split_revenue)
    print("--- Visualizations Complete ---")
    
    # =================================================================
    # TEXT REPORT
    # =================================================================
    print("\n" + "="*80)
    print("## 🚀 CAMPAIGN BUDGET ALLOCATOR AGENT REPORT (Log-NNLS Optimization)")
    print(f"Total Budget Constraint: ${total_budget:,.2f}")
    print("="*80)

    print("\n### 1. Model & Data Summary")
    print(f"* Predictive Model Quality (R-squared): **{dp.MODEL_SCORE:.4f}** (Log-NNLS Model)")
    print(f"* Overall Historical ROI (Benchmark): **{overall_hist_roi:.2f}**")
    print(f"* Total Historical Revenue/Spend: ${historical_revenue:,.2f} / ${historical_spend:,.2f}")

    print("\n### 2. Channel Benchmarks (Marginal Effectiveness)")
    print("MROI Effectiveness justifies allocation by showing the marginal revenue potential per channel.")
    print("-" * 80)
    print(recommendation_df[['Channel', 'MROI_Effectiveness']].to_string(index=False))
    print("-" * 80)

    print("\n### 3. Optimal Budget Allocation")
    print("-" * 80)
    print(recommendation_df[['Channel', 'Recommended_Budget ($)', 'Even_Split_Budget ($)']].to_string(index=False))
    print("-" * 80)

    print("\n### 4. Financial Performance Summary")
    print(summary_table.to_string())

# =========================================================================
# ENTRY POINT
# =========================================================================
if __name__ == '__main__':
    
    X_data_raw, y_data_raw, df_final_data, historical_spend, historical_sales = dp.load_and_prepare_data(dp.CSV_PATH)
    
    if X_data_raw is not None:
        X_transformed = opt.transform_features(X_data_raw)
        
        coef_dict, intercept = opt.train_sales_model(X_transformed, y_data_raw)
        
        optimal_spend, optimal_revenue, success = opt.run_budget_optimization(
            coef_dict, 
            intercept, 
            dp.TOTAL_BUDGET
        )
        
        if success:
            generate_report(
                X_data_raw, 
                df_final_data, 
                optimal_spend, 
                optimal_revenue, 
                dp.TOTAL_BUDGET
            )
        else:
            print("\n❌ Optimization Failed. The solver could not find a solution under the given constraints.")