#!/bin/bash

echo "Setting up Stock Prediction App..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3 from https://python.org"
    exit 1
fi

echo "Python found. Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing required packages..."
pip install --upgrade pip
pip install Flask==3.0.0
pip install Flask-SQLAlchemy==3.1.1
pip install Werkzeug==3.0.1
pip install yfinance==0.2.18
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.13.0
pip install statsmodels==0.14.0
pip install plotly==5.15.0
pip install python-dotenv==1.0.0
pip install gunicorn==21.2.0

echo
echo "Setup complete! To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: python app.py"
echo
