"""
Simplified version of the app for testing Flask installation
"""

try:
    from flask import Flask
    print("✅ Flask imported successfully!")
    
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return """
        <h1>🎉 Flask is working!</h1>
        <p>Your Flask installation is successful.</p>
        <p>You can now run the full stock prediction app.</p>
        """
    
    if __name__ == '__main__':
        print("Starting simple Flask test server...")
        print("Visit: http://localhost:5000")
        app.run(debug=True, port=5000)
        
except ImportError as e:
    print(f"❌ Error importing Flask: {e}")
    print("\nTo fix this issue:")
    print("1. Make sure you're in the correct directory")
    print("2. Activate your virtual environment")
    print("3. Install Flask: pip install Flask")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
