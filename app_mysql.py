from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import json
from mysql_config import Config, create_database_if_not_exists
from models_mysql import db, User, PredictionHistory, StockDataCache
from prediction import StockPredictor

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Initialize the stock predictor
predictor = StockPredictor()

def get_current_user():
    """Get current user with proper error handling"""
    if 'user_id' not in session:
        return None
    
    try:
        user = User.query.get(session['user_id'])
        if not user:
            session.clear()
            return None
        return user
    except Exception as e:
        print(f"Error getting user: {e}")
        session.clear()
        return None

def require_login():
    """Decorator to require login for routes"""
    def decorator(f):
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                if request.is_json:
                    return jsonify({'success': False, 'message': 'Please login first'})
                return redirect(url_for('login'))
            return f(user, *args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

# Routes (same as before but using MySQL)
@app.route('/')
def index():
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'No data provided'})
            
            username = data.get('username', '').strip()
            email = data.get('email', '').strip()
            password = data.get('password', '')
            
            # Validation
            if not username or not email or not password:
                return jsonify({'success': False, 'message': 'All fields are required'})
            
            if len(username) < 3:
                return jsonify({'success': False, 'message': 'Username must be at least 3 characters'})
            
            if len(password) < 6:
                return jsonify({'success': False, 'message': 'Password must be at least 6 characters'})
            
            # Check if user already exists
            if User.query.filter_by(username=username).first():
                return jsonify({'success': False, 'message': 'Username already exists'})
            
            if User.query.filter_by(email=email).first():
                return jsonify({'success': False, 'message': 'Email already exists'})
            
            # Create new user
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password_hash=hashed_password)
            
            db.session.add(new_user)
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Account created successfully'})
            
        except Exception as e:
            db.session.rollback()
            print(f"Signup error: {e}")
            return jsonify({'success': False, 'message': 'Error creating account. Please try again.'})
    
    return render_template('auth.html', mode='signup')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'No data provided'})
            
            username = data.get('username', '').strip()
            password = data.get('password', '')
            
            if not username or not password:
                return jsonify({'success': False, 'message': 'Username and password are required'})
            
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password_hash, password):
                session['user_id'] = user.id
                session['username'] = user.username
                session.permanent = True
                return jsonify({'success': True, 'message': 'Login successful'})
            else:
                return jsonify({'success': False, 'message': 'Invalid username or password'})
                
        except Exception as e:
            print(f"Login error: {e}")
            return jsonify({'success': False, 'message': 'Login error. Please try again.'})
    
    return render_template('auth.html', mode='login')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@require_login()
def dashboard(user):
    try:
        recent_predictions = PredictionHistory.query.filter_by(
            user_id=user.id
        ).order_by(
            PredictionHistory.created_at.desc()
        ).limit(5).all()
        
        return render_template('dashboard.html', user=user, recent_predictions=recent_predictions)
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        flash('Error loading dashboard. Please try again.', 'error')
        return render_template('dashboard.html', user=user, recent_predictions=[])

@app.route('/predict', methods=['POST'])
@require_login()
def predict_stock(user):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        ticker = data.get('ticker', '').upper().strip()
        model_type = data.get('model', 'linear_regression')
        days_ahead = int(data.get('days', 30))
        
        # Validation
        if not ticker:
            return jsonify({'success': False, 'message': 'Please provide a stock ticker'})
        
        if days_ahead not in [7, 14, 30, 60]:
            return jsonify({'success': False, 'message': 'Invalid prediction period'})
        
        if model_type not in ['linear_regression', 'random_forest', 'lstm', 'arima']:
            return jsonify({'success': False, 'message': 'Invalid model type'})
        
        # Get prediction from ML model
        result = predictor.predict_stock(ticker, model_type, days_ahead)
        
        if result['success']:
            # Save prediction to database
            try:
                prediction = PredictionHistory(
                    user_id=user.id,
                    ticker=ticker,
                    model_used=model_type,
                    prediction_data=json.dumps(result['data']),
                    days_predicted=days_ahead,
                    current_price=result['data'].get('current_price'),
                    predicted_price=result['data'].get('predicted_price')
                )
                db.session.add(prediction)
                db.session.commit()
            except Exception as e:
                print(f"Error saving prediction: {e}")
            
            return jsonify(result)
        else:
            return jsonify(result)
            
    except ValueError as e:
        return jsonify({'success': False, 'message': f'Invalid input: {str(e)}'})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'message': 'Error making prediction. Please try again.'})

@app.route('/api/database-stats')
@require_login()
def database_stats(user):
    """Get database statistics for admin dashboard"""
    try:
        stats = {
            'total_users': User.query.count(),
            'total_predictions': PredictionHistory.query.count(),
            'user_predictions': PredictionHistory.query.filter_by(user_id=user.id).count(),
            'popular_tickers': db.session.query(
                PredictionHistory.ticker,
                db.func.count(PredictionHistory.id).label('count')
            ).group_by(PredictionHistory.ticker).order_by(
                db.func.count(PredictionHistory.id).desc()
            ).limit(5).all(),
            'model_usage': db.session.query(
                PredictionHistory.model_used,
                db.func.count(PredictionHistory.id).label('count')
            ).group_by(PredictionHistory.model_used).order_by(
                db.func.count(PredictionHistory.id).desc()
            ).all()
        }
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({'success': False, 'message': 'Error loading statistics'})

def init_database():
    """Initialize database with error handling"""
    try:
        # Create database if it doesn't exist
        if not create_database_if_not_exists():
            print("❌ Failed to create database")
            return False
        
        with app.app_context():
            # Create all tables
            db.create_all()
            print("✅ Database tables created successfully")
            
            # Create demo user if none exist
            if User.query.count() == 0:
                demo_user = User(
                    username='demo',
                    email='demo@example.com',
                    password_hash=generate_password_hash('demo123')
                )
                db.session.add(demo_user)
                db.session.commit()
                print("✅ Demo user created (username: demo, password: demo123)")
            
            return True
                
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

if __name__ == '__main__':
    # Set session lifetime
    app.permanent_session_lifetime = timedelta(days=7)
    
    print("🗄️ XAMPP MySQL Stock Prediction App")
    print("=" * 50)
    
    # Initialize database
    if init_database():
        print("🚀 Starting Stock Prediction App with MySQL...")
        print("📱 Visit: http://localhost:5000")
        print("👤 Demo login: username=demo, password=demo123")
        print("🗄️ Database: stock_prediction_db")
        print("🌐 phpMyAdmin: http://localhost/phpmyadmin")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize database. Make sure XAMPP is running!")
