# Stock Market Prediction Frontend

A modern web interface for the Stock Market Prediction Model, built with React and Material-UI.

## Features

- Clean, intuitive user interface
- Real-time stock analysis
- Interactive price prediction charts
- Performance metrics visualization
- Technical analysis indicators
- Dark mode theme
- Responsive design

## Prerequisites

- Node.js (v14 or higher)
- Python (3.7 or higher)
- pip (Python package manager)

## Quick Start

1. Install Frontend Dependencies:
```bash
npm install
```

2. Install Backend Dependencies:
```bash
cd api
pip install -r requirements.txt
```

3. Start the Backend:
```bash
cd api
python app.py
```

4. Start the Frontend:
```bash
npm start
```

5. Open http://localhost:3003 in your browser

## Project Structure

```
stock-prediction-frontend/
├── api/                      # Flask backend
│   ├── app.py               # Main API endpoints
│   └── requirements.txt     # Python dependencies
├── public/                  # Static files
└── src/
    ├── components/          # React components
    │   ├── Header.js       # Application header
    │   ├── StockInput.js   # Stock symbol input
    │   └── ResultsDisplay.js # Analysis results
    ├── App.js              # Main application
    └── index.js            # Entry point
```

## Components

### StockInput
- Stock symbol input field
- Symbol validation
- Multi-stock support

### ResultsDisplay
- Price prediction charts
- Performance metrics table
- Technical analysis cards
- Error handling

### Header
- Application title
- Navigation elements
- Theme controls

## API Integration

### Endpoints

POST /predict
- Analyzes stocks and returns predictions
- Request body: `{ "symbols": ["AAPL", "MSFT"] }`
- Returns:
  * Price predictions
  * Performance metrics
  * Technical indicators

## Usage

1. Enter Stock Symbols:
   - Single symbol: "AAPL"
   - Multiple symbols: "AAPL,MSFT,GOOGL"

2. View Results:
   - Price prediction charts
   - Performance metrics:
     * RMSE and MAE
     * R² Score
     * Directional Accuracy
   - Technical analysis:
     * Moving Averages
     * Volume Trends
     * Price Trends

## Development

- Built with React 17+
- Material-UI for components
- Recharts for visualizations
- Axios for API requests
- Flask backend with CORS

## Error Handling

- Invalid symbol validation
- API error messages
- Loading states
- Network error handling

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
