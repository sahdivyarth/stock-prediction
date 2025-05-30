#!/usr/bin/env python3
"""
Test MySQL connection for XAMPP
"""

import pymysql
import sys

def test_connection():
    """Test basic MySQL connection"""
    print("🔍 Testing XAMPP MySQL Connection...")
    print("=" * 40)
    
    try:
        # XAMPP default settings
        connection = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='',  # Default XAMPP has no password
            charset='utf8mb4'
        )
        
        cursor = connection.cursor()
        
        # Test basic queries
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        
        cursor.execute("SELECT USER()")
        user = cursor.fetchone()
        
        print("✅ Connection successful!")
        print(f"📊 MySQL Version: {version[0]}")
        print(f"👤 Connected as: {user[0]}")
        print(f"🗄️  Available databases: {len(databases)}")
        
        # List some databases
        print("\n📋 Databases:")
        for db in databases[:5]:  # Show first 5
            print(f"   • {db[0]}")
        if len(databases) > 5:
            print(f"   ... and {len(databases) - 5} more")
        
        cursor.close()
        connection.close()
        
        print("\n🎉 XAMPP MySQL is working correctly!")
        return True
        
    except pymysql.Error as e:
        print(f"❌ MySQL Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Open XAMPP Control Panel")
        print("2. Start MySQL service (click Start button)")
        print("3. Ensure MySQL shows green status")
        print("4. Check if port 3306 is available")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if test_connection():
        sys.exit(0)
    else:
        sys.exit(1)
