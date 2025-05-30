"""
Quick test to verify the predictor works correctly
"""

from fixed_robust_predictor import FixedRobustStockPredictor

def test_predictor():
    print("🧪 Testing Fixed Robust Stock Predictor")
    print("=" * 50)
    
    predictor = FixedRobustStockPredictor()
    
    # Test with SPOT (the ticker that was failing)
    test_tickers = ['SPOT', 'AAPL', 'GOOGL', 'TSLA', 'INVALID']
    
    for ticker in test_tickers:
        print(f"\n📊 Testing {ticker}:")
        print("-" * 30)
        
        try:
            result = predictor.predict_stock(ticker, 'linear_regression', 30)
            
            if result['success']:
                data = result['data']
                print(f"✅ SUCCESS!")
                print(f"   Current price: ${data['current_price']:.2f}")
                print(f"   30-day prediction: ${data['predictions'][-1]:.2f}")
                print(f"   Change: {((data['predictions'][-1] - data['current_price']) / data['current_price'] * 100):+.1f}%")
                print(f"   Data source: {data['data_source']}")
                
                if 'note' in data:
                    print(f"   📝 {data['note']}")
            else:
                print(f"❌ FAILED: {result['message']}")
                
        except Exception as e:
            print(f"💥 EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_predictor()
