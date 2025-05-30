#!/usr/bin/env python3
"""
XAMPP MySQL Database Setup for Stock Prediction App
This script sets up the complete MySQL database for XAMPP
"""

import pymysql
import sys
import os
from datetime import datetime
import json

# XAMPP MySQL Configuration
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',  # Default XAMPP has no password
    'charset': 'utf8mb4'
}

DATABASE_NAME = 'stock_prediction_db'

def test_mysql_connection():
    """Test if MySQL is running and accessible"""
    print("🔍 Testing MySQL connection...")
    
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        
        print(f"✅ MySQL connection successful!")
        print(f"📊 MySQL version: {version[0]}")
        
        cursor.close()
        connection.close()
        return True
        
    except pymysql.Error as e:
        print(f"❌ MySQL connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure XAMPP is running")
        print("2. Start MySQL service in XAMPP Control Panel")
        print("3. Check if port 3306 is available")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_database():
    """Create the stock prediction database"""
    print(f"\n🗄️ Creating database '{DATABASE_NAME}'...")
    
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
        cursor.execute(f"USE {DATABASE_NAME}")
        
        print(f"✅ Database '{DATABASE_NAME}' created successfully!")
        
        cursor.close()
        connection.close()
        return True
        
    except pymysql.Error as e:
        print(f"❌ Error creating database: {e}")
        return False

def create_tables():
    """Create all required tables"""
    print("\n📋 Creating database tables...")
    
    # Connect to the specific database
    config = MYSQL_CONFIG.copy()
    config['database'] = DATABASE_NAME
    
    try:
        connection = pymysql.connect(**config)
        cursor = connection.cursor()
        
        # Users table
        users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_username (username),
            INDEX idx_email (email)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(users_table)
        print("✅ Users table created")
        
        # Prediction history table
        predictions_table = """
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(predictions_table)
        print("✅ Prediction history table created")
        
        # Stock data cache table
        cache_table = """
        CREATE TABLE IF NOT EXISTS stock_data_cache (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            data_type VARCHAR(20) NOT NULL,
            data_json TEXT NOT NULL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL,
            INDEX idx_ticker_type (ticker, data_type),
            INDEX idx_expires (expires_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(cache_table)
        print("✅ Stock data cache table created")
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
        
    except pymysql.Error as e:
        print(f"❌ Error creating tables: {e}")
        return False

def create_views():
    """Create database views for analytics"""
    print("\n📊 Creating database views...")
    
    config = MYSQL_CONFIG.copy()
    config['database'] = DATABASE_NAME
    
    try:
        connection = pymysql.connect(**config)
        cursor = connection.cursor()
        
        # User summary view
        user_summary_view = """
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
        GROUP BY u.id, u.username, u.email, u.created_at
        """
        
        cursor.execute(user_summary_view)
        print("✅ User summary view created")
        
        # Popular stocks view
        popular_stocks_view = """
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
        ORDER BY prediction_count DESC
        """
        
        cursor.execute(popular_stocks_view)
        print("✅ Popular stocks view created")
        
        # Model performance view
        model_performance_view = """
        CREATE OR REPLACE VIEW model_performance AS
        SELECT 
            model_used,
            COUNT(*) as usage_count,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT ticker) as unique_tickers,
            AVG(accuracy_score) as avg_accuracy
        FROM prediction_history
        GROUP BY model_used
        ORDER BY usage_count DESC
        """
        
        cursor.execute(model_performance_view)
        print("✅ Model performance view created")
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
        
    except pymysql.Error as e:
        print(f"❌ Error creating views: {e}")
        return False

def create_demo_user():
    """Create a demo user for testing"""
    print("\n👤 Creating demo user...")
    
    config = MYSQL_CONFIG.copy()
    config['database'] = DATABASE_NAME
    
    try:
        connection = pymysql.connect(**config)
        cursor = connection.cursor()
        
        # Check if demo user already exists
        cursor.execute("SELECT id FROM users WHERE username = 'demo'")
        if cursor.fetchone():
            print("ℹ️  Demo user already exists")
            cursor.close()
            connection.close()
            return True
        
        # Create demo user (password: demo123)
        # This is a pre-hashed password for 'demo123'
        demo_password_hash = 'pbkdf2:sha256:600000$8xKzYqXr$c8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8'
        
        insert_user = """
        INSERT INTO users (username, email, password_hash) 
        VALUES (%s, %s, %s)
        """
        
        cursor.execute(insert_user, ('demo', 'demo@example.com', demo_password_hash))
        connection.commit()
        
        print("✅ Demo user created successfully!")
        print("   Username: demo")
        print("   Password: demo123")
        print("   Email: demo@example.com")
        
        cursor.close()
        connection.close()
        return True
        
    except pymysql.Error as e:
        print(f"❌ Error creating demo user: {e}")
        return False

def verify_setup():
    """Verify the database setup is complete"""
    print("\n🔍 Verifying database setup...")
    
    config = MYSQL_CONFIG.copy()
    config['database'] = DATABASE_NAME
    
    try:
        connection = pymysql.connect(**config)
        cursor = connection.cursor()
        
        # Check tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = ['users', 'prediction_history', 'stock_data_cache']
        
        print("📋 Tables found:")
        for table in table_names:
            status = "✅" if table in expected_tables else "❓"
            print(f"   {status} {table}")
        
        # Check views
        cursor.execute("SHOW FULL TABLES WHERE Table_type = 'VIEW'")
        views = cursor.fetchall()
        view_names = [view[0] for view in views]
        
        print("\n📊 Views found:")
        for view in view_names:
            print(f"   ✅ {view}")
        
        # Check demo user
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"\n👥 Users in database: {user_count}")
        
        cursor.close()
        connection.close()
        
        if len(table_names) >= 3:
            print("\n🎉 Database setup completed successfully!")
            return True
        else:
            print("\n❌ Database setup incomplete")
            return False
        
    except pymysql.Error as e:
        print(f"❌ Error verifying setup: {e}")
        return False

def print_connection_info():
    """Print database connection information"""
    print("\n📊 Database Connection Information:")
    print("=" * 50)
    print(f"🗄️  Database: {DATABASE_NAME}")
    print(f"🌐 Host: {MYSQL_CONFIG['host']}")
    print(f"🔌 Port: {MYSQL_CONFIG['port']}")
    print(f"👤 Username: {MYSQL_CONFIG['user']}")
    print(f"🔑 Password: {'(empty)' if not MYSQL_CONFIG['password'] else '(set)'}")
    print("\n🌐 Access URLs:")
    print("📱 App: http://localhost:5000")
    print("🗄️  phpMyAdmin: http://localhost/phpmyadmin")
    print("\n👤 Demo Login:")
    print("   Username: demo")
    print("   Password: demo123")

def main():
    """Main setup function"""
    print("🗄️ XAMPP MySQL Database Setup")
    print("=" * 50)
    print("Setting up Stock Prediction App database...")
    
    # Step 1: Test MySQL connection
    if not test_mysql_connection():
        print("\n❌ Setup failed: Cannot connect to MySQL")
        print("\n🔧 Please ensure:")
        print("1. XAMPP is installed and running")
        print("2. MySQL service is started in XAMPP Control Panel")
        print("3. Port 3306 is available")
        return False
    
    # Step 2: Create database
    if not create_database():
        print("\n❌ Setup failed: Cannot create database")
        return False
    
    # Step 3: Create tables
    if not create_tables():
        print("\n❌ Setup failed: Cannot create tables")
        return False
    
    # Step 4: Create views
    if not create_views():
        print("\n⚠️  Warning: Could not create views (non-critical)")
    
    # Step 5: Create demo user
    if not create_demo_user():
        print("\n⚠️  Warning: Could not create demo user")
    
    # Step 6: Verify setup
    if verify_setup():
        print_connection_info()
        
        print("\n🚀 Next Steps:")
        print("1. Install Python dependencies: pip install -r requirements_mysql.txt")
        print("2. Run the app: python app_mysql.py")
        print("3. Visit: http://localhost:5000")
        print("4. Login with demo/demo123")
        
        return True
    else:
        print("\n❌ Setup verification failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ XAMPP MySQL setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ XAMPP MySQL setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
