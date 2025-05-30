import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# MySQL Database Configuration for XAMPP
class Config:
    # XAMPP MySQL default settings
    MYSQL_HOST = 'localhost'
    MYSQL_PORT = 3306
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = ''  # Default XAMPP has no password
    MYSQL_DATABASE = 'stock_prediction_db'
    
    # Flask-SQLAlchemy URI
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'your-secret-key-change-in-production'

def create_database_if_not_exists():
    """Create database if it doesn't exist"""
    import pymysql
    
    try:
        # Connect to MySQL server (without specifying database)
        connection = pymysql.connect(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD
        )
        
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DATABASE}")
        cursor.execute(f"USE {Config.MYSQL_DATABASE}")
        
        print(f"✅ Database '{Config.MYSQL_DATABASE}' created/verified successfully")
        
        cursor.close()
        connection.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        print("🔧 Make sure XAMPP MySQL is running!")
        return False

if __name__ == "__main__":
    print("🗄️ XAMPP MySQL Database Setup")
    print("=" * 40)
    
    # Test database creation
    if create_database_if_not_exists():
        print("✅ Ready to use MySQL database with XAMPP!")
    else:
        print("❌ Database setup failed. Check XAMPP status.")
