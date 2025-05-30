#!/usr/bin/env python3
"""
Create the stock prediction database in XAMPP MySQL
"""

import pymysql
import sys

def create_database():
    """Create the stock_prediction_db database"""
    print("🗄️ Creating Stock Prediction Database...")
    print("=" * 45)
    
    try:
        # Connect to MySQL server
        connection = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='',
            charset='utf8mb4'
        )
        
        cursor = connection.cursor()
        
        # Create database
        database_name = 'stock_prediction_db'
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        
        # Verify database was created
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        
        if database_name in databases:
            print(f"✅ Database '{database_name}' created successfully!")
            
            # Use the database
            cursor.execute(f"USE {database_name}")
            
            # Show current database
            cursor.execute("SELECT DATABASE()")
            current_db = cursor.fetchone()
            print(f"📊 Current database: {current_db[0]}")
            
        else:
            print(f"❌ Failed to create database '{database_name}'")
            return False
        
        cursor.close()
        connection.close()
        
        print("\n🎉 Database creation completed!")
        print(f"🌐 Access via phpMyAdmin: http://localhost/phpmyadmin")
        print(f"🗄️  Database name: {database_name}")
        
        return True
        
    except pymysql.Error as e:
        print(f"❌ MySQL Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if create_database():
        sys.exit(0)
    else:
        sys.exit(1)
