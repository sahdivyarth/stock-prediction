"""
Examples of different chart types available in the app
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_charts():
    """Create sample charts showing different visualization types"""
    
    print("📊 CREATING SAMPLE PREDICTION CHARTS")
    print("=" * 50)
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    historical_dates = dates[:300]  # 10 months historical
    future_dates = dates[300:330]   # 1 month predictions
    
    # Sample historical prices
    np.random.seed(42)
    base_price = 150
    historical_prices = []
    current_price = base_price
    
    for i in range(len(historical_dates)):
        change = np.random.normal(0.001, 0.02)  # Daily change
        current_price *= (1 + change)
        historical_prices.append(current_price)
    
    # Sample predictions
    predictions = {
        'Linear Regression': [],
        'Random Forest': [],
        'LSTM': [],
        'ARIMA': []
    }
    
    for model in predictions.keys():
        pred_prices = []
        last_price = historical_prices[-1]
        
        for i in range(len(future_dates)):
            if model == 'Linear Regression':
                change = np.random.normal(0.002, 0.015)
            elif model == 'Random Forest':
                change = np.random.normal(0.0015, 0.018)
            elif model == 'LSTM':
                change = np.random.normal(0.0025, 0.012)
            else:  # ARIMA
                change = np.random.normal(0.001, 0.016)
            
            last_price *= (1 + change)
            pred_prices.append(last_price)
        
        predictions[model] = pred_prices
    
    # 1. Main Prediction Chart
    create_main_prediction_chart(historical_dates, historical_prices, 
                               future_dates, predictions)
    
    # 2. Model Comparison Chart
    create_model_comparison_chart(future_dates, predictions)
    
    # 3. Technical Analysis Chart
    create_technical_chart(historical_dates, historical_prices)
    
    # 4. Performance Metrics Chart
    create_metrics_chart()
    
    print("\n✅ All sample charts created successfully!")
    print("📁 Check the generated HTML files to see the interactive charts")

def create_main_prediction_chart(hist_dates, hist_prices, future_dates, predictions):
    """Create the main prediction chart"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_prices,
        mode='lines',
        name='Historical Prices',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Predictions
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    dash_styles = ['solid', 'dash', 'dot', 'dashdot']
    
    for i, (model, pred_prices) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred_prices,
            mode='lines+markers',
            name=f'{model}',
            line=dict(color=colors[i], width=2, dash=dash_styles[i]),
            marker=dict(size=6),
            hovertemplate=f'{model}<br>Date: %{{x}}<br>Predicted: $%{{y:.2f}}<extra></extra>'
        ))
    
    # Add vertical line to separate historical and predicted data
    fig.add_vline(
        x=hist_dates[-1], 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Prediction Start"
    )
    
    fig.update_layout(
        title='Stock Price Prediction - Multiple ML Models',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.write_html('main_prediction_chart.html')
    print("✅ Main prediction chart created: main_prediction_chart.html")

def create_model_comparison_chart(future_dates, predictions):
    """Create model comparison chart"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(predictions.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (model, pred_prices) in enumerate(predictions.items()):
        row, col = positions[i]
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=pred_prices,
                mode='lines+markers',
                name=model,
                line=dict(color=colors[i], width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add trend line
        x_numeric = list(range(len(pred_prices)))
        z = np.polyfit(x_numeric, pred_prices, 1)
        trend_line = np.poly1d(z)(x_numeric)
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=trend_line,
                mode='lines',
                line=dict(color='red', dash='dash', width=1),
                name='Trend',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Model Comparison - Individual Predictions',
        height=600,
        template='plotly_white'
    )
    
    fig.write_html('model_comparison_chart.html')
    print("✅ Model comparison chart created: model_comparison_chart.html")

def create_technical_chart(hist_dates, hist_prices):
    """Create technical analysis chart"""
    
    # Calculate technical indicators
    df = pd.DataFrame({'Close': hist_prices}, index=hist_dates)
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price & Technical Indicators', 'RSI'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name='Close Price',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA_20'], name='20-day MA',
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA_50'], name='50-day MA',
        line=dict(color='red', width=1)
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_upper'], name='BB Upper',
        line=dict(color='gray', width=1, dash='dash'),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_lower'], name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='purple', width=2)
    ), row=2, col=1)
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        title='Technical Analysis with Indicators',
        height=700,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    fig.write_html('technical_analysis_chart.html')
    print("✅ Technical analysis chart created: technical_analysis_chart.html")

def create_metrics_chart():
    """Create model performance metrics chart"""
    
    models = ['Linear Regression', 'Random Forest', 'LSTM', 'ARIMA']
    mse_values = [2.45, 1.89, 1.67, 2.12]
    mae_values = [1.23, 1.05, 0.98, 1.15]
    accuracy_values = [85.2, 87.8, 89.1, 86.5]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Mean Squared Error', 'Mean Absolute Error', 'Accuracy %'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # MSE
    fig.add_trace(go.Bar(
        x=models, y=mse_values, name='MSE',
        marker_color='lightcoral'
    ), row=1, col=1)
    
    # MAE
    fig.add_trace(go.Bar(
        x=models, y=mae_values, name='MAE',
        marker_color='lightblue'
    ), row=1, col=2)
    
    # Accuracy
    fig.add_trace(go.Bar(
        x=models, y=accuracy_values, name='Accuracy',
        marker_color='lightgreen'
    ), row=1, col=3)
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.write_html('performance_metrics_chart.html')
    print("✅ Performance metrics chart created: performance_metrics_chart.html")

if __name__ == "__main__":
    create_sample_charts()
    
    print(f"\n🎯 CHART FEATURES SUMMARY:")
    print("=" * 40)
    print("✅ Interactive zoom, pan, hover")
    print("✅ Multiple model predictions")
    print("✅ Technical indicators overlay")
    print("✅ Performance metrics visualization")
    print("✅ Export as PNG/HTML/PDF")
    print("✅ Real-time data updates")
    print("✅ Mobile-responsive design")
    
    print(f"\n📱 To see these charts in the live app:")
    print("   python app.py")
