import time
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data_fetching import get_stock_data
from utils.data_analysis_and_visualization import main_analysis, plot_correlation_analysis
from utils.risk_analysis import analyze_risk
from model_training.model_training_and_prediction import predict_stock_price, generate_report, generate_final_report

def get_valid_stock_symbols():
    """Get valid stock symbols from user input"""
    while True:
        tech_list = input("Enter the stock symbols (separated by spaces): ").split()
        if not tech_list:
            print("Error: Please enter at least one stock symbol.")
            continue
        # Remove any empty strings and whitespace
        tech_list = [symbol.strip() for symbol in tech_list if symbol.strip()]
        if not tech_list:
            print("Error: Please enter valid stock symbols.")
            continue
        return tech_list

def main():
    """Main function to run the analysis"""
    start_time = time.time()
    
    # Ask user for stock names with validation
    tech_list = get_valid_stock_symbols()
    
    # Get stock data
    df, company_list, _ = get_stock_data(tech_list)
    
    # Perform main analysis
    main_analysis(company_list, tech_list)
    
    # Correlation analysis
    tech_rets = plot_correlation_analysis(tech_list)
    
    # Risk analysis
    analyze_risk(tech_rets)
    
    # Dictionary to store metrics for all stocks
    all_metrics = {}
    
    # Predict stock prices for each symbol and generate report
    for symbol in tech_list:
        predictions, metrics = predict_stock_price(symbol)
        print(f"\nPredicted vs Actual Prices for {symbol}:")
        print(predictions.tail(30))  # Show last 30 days
        generate_report(symbol, predictions, metrics)
        all_metrics[symbol] = metrics
    
    # Generate final comprehensive report
    generate_final_report(all_metrics)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print("\nAll reports have been generated in the results directory.")
    print("Individual stock reports and prediction plots are saved as [SYMBOL]_prediction_report.txt and [SYMBOL]_prediction_plot.png")
    print("A comprehensive final report has been saved as final_model_report.txt")
    
    plt.show()  # Ensure all plots are displayed

if __name__ == "__main__":
    main()
