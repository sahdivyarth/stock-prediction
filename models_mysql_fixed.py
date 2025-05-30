from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Numeric

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with predictions
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'total_predictions': len(self.predictions)
        }

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    ticker = db.Column(db.String(10), nullable=False, index=True)
    model_used = db.Column(db.String(50), nullable=False, index=True)
    prediction_data = db.Column(db.Text, nullable=False)  # JSON string
    days_predicted = db.Column(db.Integer, nullable=False)
    current_price = db.Column(Numeric(10, 2), nullable=True)
    predicted_price = db.Column(Numeric(10, 2), nullable=True)
    accuracy_score = db.Column(Numeric(5, 4), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f'<Prediction {self.ticker} - {self.model_used}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else 'Unknown',
            'ticker': self.ticker,
            'model_used': self.model_used,
            'days_predicted': self.days_predicted,
            'current_price': float(self.current_price) if self.current_price else None,
            'predicted_price': float(self.predicted_price) if self.predicted_price else None,
            'accuracy_score': float(self.accuracy_score) if self.accuracy_score else None,
            'created_at': self.created_at.isoformat()
        }

# Additional model for storing stock data cache
class StockDataCache(db.Model):
    __tablename__ = 'stock_data_cache'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String(10), nullable=False, index=True)
    data_type = db.Column(db.String(20), nullable=False)  # 'historical', 'realtime'
    data_json = db.Column(db.Text, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    def __repr__(self):
        return f'<StockCache {self.ticker} - {self.data_type}>'
