# Stock Market Prediction App

A comprehensive full-stack web application built with Flask that predicts stock market prices using multiple machine learning models. The app features an interactive frontend UI, robust backend integration, and secure user authentication.

## 🚀 Features

### Core Features
- **User Authentication**: Secure signup, login, and logout functionality with password hashing
- **Stock Market Prediction**: Support for multiple ML models (Linear Regression, Random Forest, LSTM, ARIMA)
- **Interactive Charts**: Beautiful visualizations using Plotly.js
- **Model Comparison**: Compare performance metrics across different models
- **Prediction History**: Track and view past predictions
- **Responsive Design**: Modern UI built with Bootstrap 5

### Machine Learning Models
1. **Linear Regression**: Fast and interpretable baseline model
2. **Random Forest**: Ensemble method for robust predictions
3. **LSTM Neural Network**: Deep learning for complex patterns
4. **ARIMA**: Time series analysis for trend-based predictions

## 🛠️ Technology Stack

- **Backend**: Flask, SQLAlchemy, SQLite
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: Scikit-learn, TensorFlow/Keras, Statsmodels
- **Data**: Yahoo Finance API (yfinance)
- **Visualization**: Plotly.js
- **Authentication**: Flask sessions with Werkzeug password hashing

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/yourusername/stock-prediction-app.git
   cd stock-prediction-app
   \`\`\`

2. **Create a virtual environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   \`\`\`

3. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Set up environment variables** (optional)
   \`\`\`bash
   cp .env.example .env
   # Edit .env with your configuration
   \`\`\`

5. **Initialize the database**
   \`\`\`bash
   python app.py
   \`\`\`

6. **Run the application**
   \`\`\`bash
   python app.py
   \`\`\`

The application will be available at \`http://localhost:5000\`

## 🎯 Usage

### Getting Started
1. **Sign Up**: Create a new account or log in with existing credentials
2. **Dashboard**: Access the main prediction interface
3. **Make Predictions**: 
   - Enter a stock ticker symbol (e.g., AAPL, GOOGL)
   - Select your preferred ML model
   - Choose prediction timeframe (7-60 days)
   - Click "Predict" to generate forecasts

### Model Comparison
- Use the "Compare Models" feature to evaluate different algorithms
- View performance metrics (MSE, MAE) for each model
- Choose the best-performing model for your predictions

### Viewing History
- Access your prediction history from the navigation menu
- Review past predictions and their accuracy
- Analyze model performance over time

## 📊 Model Performance

### Metrics Explained
- **MSE (Mean Squared Error)**: Lower values indicate better accuracy
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **Model Comparison**: Side-by-side performance analysis

### Best Practices
- **LSTM**: Best for volatile stocks with complex patterns
- **Random Forest**: Good balance of accuracy and speed
- **Linear Regression**: Fast baseline for stable stocks
- **ARIMA**: Effective for trend-following strategies

## 🏗️ Project Structure

\`\`\`
stock-prediction-app/
├── app.py                 # Main Flask application
├── models.py              # Database models
├── prediction.py          # ML prediction logic
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── main.js       # Frontend JavaScript
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # Landing page
│   ├── auth.html         # Login/signup
│   ├── dashboard.html    # Main dashboard
│   └── history.html      # Prediction history
└── instance/
    └── stock_app.db      # SQLite database
\`\`\`

## 🔧 Configuration

### Environment Variables
Create a \`.env\` file in the root directory:

\`\`\`env
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///stock_app.db
FLASK_ENV=development
\`\`\`

### Database Configuration
The app uses SQLite by default. To use PostgreSQL:

\`\`\`python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/dbname'
\`\`\`

## 🚀 Deployment

### Heroku Deployment
1. **Create Heroku app**
   \`\`\`bash
   heroku create your-app-name
   \`\`\`

2. **Set environment variables**
   \`\`\`bash
   heroku config:set SECRET_KEY=your-secret-key
   \`\`\`

3. **Deploy**
   \`\`\`bash
   git push heroku main
   \`\`\`

### Docker Deployment
\`\`\`dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
\`\`\`

## 🧪 Testing

### Running Tests
\`\`\`bash
python -m pytest tests/
\`\`\`

### Model Validation
- Backtesting on historical data
- Cross-validation for model selection
- Performance monitoring in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (\`git checkout -b feature/amazing-feature\`)
3. Commit your changes (\`git commit -m 'Add amazing feature'\`)
4. Push to the branch (\`git push origin feature/amazing-feature\`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/stock-prediction-app/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock data API
- Scikit-learn and TensorFlow communities
- Bootstrap and Plotly.js for UI components
- Flask community for excellent documentation

## 📈 Future Enhancements

- [ ] Real-time stock data streaming
- [ ] Advanced technical indicators
- [ ] Portfolio optimization features
- [ ] Mobile app development
- [ ] API for external integrations
- [ ] Advanced charting tools
- [ ] Social features and community predictions
