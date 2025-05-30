-- Stock Prediction Database Schema for XAMPP MySQL
-- Run this in phpMyAdmin if needed

CREATE DATABASE IF NOT EXISTS stock_prediction_db;
USE stock_prediction_db;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

-- Prediction history table
CREATE TABLE IF NOT EXISTS prediction_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    prediction_data TEXT NOT NULL,
    days_predicted INT NOT NULL,
    current_price DECIMAL(10,2) NULL,
    predicted_price DECIMAL(10,2) NULL,
    accuracy_score DECIMAL(5,4) NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_ticker (ticker),
    INDEX idx_model (model_used),
    INDEX idx_created_at (created_at)
);

-- Stock data cache table
CREATE TABLE IF NOT EXISTS stock_data_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    data_type VARCHAR(20) NOT NULL,
    data_json TEXT NOT NULL,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    INDEX idx_ticker_type (ticker, data_type),
    INDEX idx_expires (expires_at)
);

-- Insert demo user
INSERT IGNORE INTO users (username, email, password_hash) 
VALUES ('demo', 'demo@example.com', 'pbkdf2:sha256:600000$...');

-- Create views for easy data access
CREATE OR REPLACE VIEW user_prediction_summary AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.created_at as user_created,
    COUNT(p.id) as total_predictions,
    COUNT(DISTINCT p.ticker) as unique_tickers,
    MAX(p.created_at) as last_prediction
FROM users u
LEFT JOIN prediction_history p ON u.id = p.user_id
GROUP BY u.id, u.username, u.email, u.created_at;

CREATE OR REPLACE VIEW popular_stocks AS
SELECT 
    ticker,
    COUNT(*) as prediction_count,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(current_price) as avg_current_price,
    AVG(predicted_price) as avg_predicted_price,
    MAX(created_at) as last_predicted
FROM prediction_history
GROUP BY ticker
ORDER BY prediction_count DESC;

CREATE OR REPLACE VIEW model_performance AS
SELECT 
    model_used,
    COUNT(*) as usage_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT ticker) as unique_tickers,
    AVG(accuracy_score) as avg_accuracy
FROM prediction_history
GROUP BY model_used
ORDER BY usage_count DESC;
