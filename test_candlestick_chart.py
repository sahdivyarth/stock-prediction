import pandas as pd
from utils.data_analysis_and_visualization import plot_candlestick_charts

# Create a test DataFrame with different column names
data = {
    'date': pd.date_range('2023-01-01', periods=10),
    'open': [100, 101, 99, 102, 98, 103, 97, 104, 96,105],
    'high': [102, 103, 101, 104, 100, 105, 99, 106, 98, 107],
    'low': [98, 99, 97, 100, 96, 101, 95, 102, 94, 103],
    'close': [101, 100, 100, 101, 99, 102, 98, 103, 97, 104],
    'adj_close': [100.5, 99.5, 99.5, 100.5, 98.5, 101.5, 97.5, 102.5, 96.5, 103.5],
    'volume': [1000, 2000, 1500, 2500, 1200, 3000, 1800, 2800, 1600, 3200],
    'company': ['ACME Inc.'] * 10
}

test_df = pd.DataFrame(data)

# Call the plot_candlestick_charts function with the test DataFrame
plot_candlestick_charts([test_df], ['ACME Inc.'])
