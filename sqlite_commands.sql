-- SQLite Command Line Reference for Stock Prediction App
-- Copy and paste these commands into the SQLite prompt

-- ============================================
-- BASIC SQLITE COMMANDS (start with a dot)
-- ============================================

-- Show all tables in the database
.tables

-- Show database schema (table structures)
.schema

-- Show column headers in query results
.headers on

-- Show results in a nice table format
.mode table

-- Show current settings
.show

-- Get help
.help

-- Exit SQLite
.quit

-- ============================================
-- VIEW DATABASE STRUCTURE
-- ============================================

-- See the structure of the user table
.schema user

-- See the structure of the prediction_history table
.schema prediction_history

-- ============================================
-- VIEW ALL DATA
-- ============================================

-- View all users
SELECT * FROM user;

-- View all predictions
SELECT * FROM prediction_history;

-- Count total users
SELECT COUNT(*) as total_users FROM user;

-- Count total predictions
SELECT COUNT(*) as total_predictions FROM prediction_history;

-- ============================================
-- USER QUERIES
-- ============================================

-- View users with their creation dates
SELECT 
    id,
    username,
    email,
    created_at
FROM user
ORDER BY created_at DESC;

-- Find a specific user
SELECT * FROM user WHERE username = 'demo';

-- ============================================
-- PREDICTION QUERIES
-- ============================================

-- View recent predictions with user info
SELECT 
    p.id,
    u.username,
    p.ticker,
    p.model_used,
    p.days_predicted,
    p.created_at
FROM prediction_history p
JOIN user u ON p.user_id = u.id
ORDER BY p.created_at DESC
LIMIT 10;

-- Find predictions for a specific stock
SELECT 
    u.username,
    p.ticker,
    p.model_used,
    p.created_at
FROM prediction_history p
JOIN user u ON p.user_id = u.id
WHERE p.ticker = 'AAPL'
ORDER BY p.created_at DESC;

-- Count predictions by model
SELECT 
    model_used,
    COUNT(*) as count
FROM prediction_history
GROUP BY model_used
ORDER BY count DESC;

-- Count predictions by user
SELECT 
    u.username,
    COUNT(p.id) as prediction_count
FROM user u
LEFT JOIN prediction_history p ON u.id = p.user_id
GROUP BY u.username
ORDER BY prediction_count DESC;

-- ============================================
-- ADVANCED QUERIES
-- ============================================

-- Most popular stocks
SELECT 
    ticker,
    COUNT(*) as prediction_count
FROM prediction_history
GROUP BY ticker
ORDER BY prediction_count DESC
LIMIT 10;

-- User activity by date
SELECT 
    DATE(created_at) as date,
    COUNT(*) as predictions
FROM prediction_history
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Model usage statistics
SELECT 
    model_used,
    COUNT(*) as times_used,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM prediction_history), 2) as percentage
FROM prediction_history
GROUP BY model_used
ORDER BY times_used DESC;

-- ============================================
-- DATA EXPORT COMMANDS
-- ============================================

-- Export users to CSV
.headers on
.mode csv
.output users_export.csv
SELECT * FROM user;
.output stdout

-- Export predictions to CSV
.output predictions_export.csv
SELECT 
    p.id,
    u.username,
    p.ticker,
    p.model_used,
    p.days_predicted,
    p.created_at
FROM prediction_history p
JOIN user u ON p.user_id = u.id;
.output stdout

-- Reset to table mode
.mode table

-- ============================================
-- DATABASE MAINTENANCE
-- ============================================

-- Check database integrity
PRAGMA integrity_check;

-- Get database info
PRAGMA database_list;

-- Show database size info
.dbinfo

-- Vacuum database (optimize)
VACUUM;

-- ============================================
-- BACKUP COMMANDS (run from command line, not in SQLite)
-- ============================================

-- Create backup:
-- sqlite3 stock_app.db ".backup backup_$(date +%Y%m%d).db"

-- Restore from backup:
-- sqlite3 new_stock_app.db ".restore backup_20241201.db"
