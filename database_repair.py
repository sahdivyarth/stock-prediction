"""
Database repair and maintenance script
"""

import sqlite3
import os
from datetime import datetime

def repair_database():
    """Repair and clean up the database"""
    
    db_path = 'stock_app.db'
    backup_path = f'stock_app_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
    
    print("🔧 DATABASE REPAIR UTILITY")
    print("=" * 40)
    
    # Check if database exists
    if not os.path.exists(db_path):
        print("❌ Database file not found. Creating new database...")
        create_fresh_database()
        return
    
    try:
        # Create backup
        print(f"📋 Creating backup: {backup_path}")
        import shutil
        shutil.copy2(db_path, backup_path)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Found tables: {[table[0] for table in tables]}")
        
        # Check user table
        try:
            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            print(f"👥 Users in database: {user_count}")
            
            # Check for orphaned sessions
            cursor.execute("SELECT id, username FROM user")
            users = cursor.fetchall()
            
            if users:
                print("✅ User table is healthy")
                for user_id, username in users:
                    print(f"   User ID {user_id}: {username}")
            else:
                print("⚠️  No users found. Creating demo user...")
                create_demo_user(cursor)
                
        except sqlite3.Error as e:
            print(f"❌ User table error: {e}")
            print("🔨 Recreating user table...")
            recreate_user_table(cursor)
        
        # Check prediction_history table
        try:
            cursor.execute("SELECT COUNT(*) FROM prediction_history")
            prediction_count = cursor.fetchone()[0]
            print(f"📈 Predictions in database: {prediction_count}")
            
        except sqlite3.Error as e:
            print(f"❌ Prediction history table error: {e}")
            print("🔨 Recreating prediction_history table...")
            recreate_prediction_table(cursor)
        
        # Clean up orphaned records
        cleanup_orphaned_records(cursor)
        
        conn.commit()
        conn.close()
        
        print("✅ Database repair completed successfully!")
        
    except Exception as e:
        print(f"❌ Database repair failed: {e}")
        print("🔄 Creating fresh database...")
        create_fresh_database()

def create_demo_user(cursor):
    """Create a demo user"""
    from werkzeug.security import generate_password_hash
    
    hashed_password = generate_password_hash('demo123')
    cursor.execute("""
        INSERT INTO user (username, email, password_hash, created_at)
        VALUES (?, ?, ?, ?)
    """, ('demo', 'demo@example.com', hashed_password, datetime.utcnow()))
    
    print("✅ Demo user created (username: demo, password: demo123)")

def recreate_user_table(cursor):
    """Recreate the user table"""
    cursor.execute("DROP TABLE IF EXISTS user")
    cursor.execute("""
        CREATE TABLE user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    create_demo_user(cursor)

def recreate_prediction_table(cursor):
    """Recreate the prediction_history table"""
    cursor.execute("DROP TABLE IF EXISTS prediction_history")
    cursor.execute("""
        CREATE TABLE prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            model_used VARCHAR(50) NOT NULL,
            prediction_data TEXT NOT NULL,
            days_predicted INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user (id)
        )
    """)

def cleanup_orphaned_records(cursor):
    """Clean up orphaned prediction records"""
    cursor.execute("""
        DELETE FROM prediction_history 
        WHERE user_id NOT IN (SELECT id FROM user)
    """)
    
    deleted = cursor.rowcount
    if deleted > 0:
        print(f"🧹 Cleaned up {deleted} orphaned prediction records")

def create_fresh_database():
    """Create a completely fresh database"""
    
    print("🆕 Creating fresh database...")
    
    # Remove old database if it exists
    if os.path.exists('stock_app.db'):
        os.remove('stock_app.db')
    
    # Create new database
    conn = sqlite3.connect('stock_app.db')
    cursor = conn.cursor()
    
    # Create user table
    cursor.execute("""
        CREATE TABLE user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create prediction_history table
    cursor.execute("""
        CREATE TABLE prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            model_used VARCHAR(50) NOT NULL,
            prediction_data TEXT NOT NULL,
            days_predicted INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user (id)
        )
    """)
    
    # Create demo user
    create_demo_user(cursor)
    
    conn.commit()
    conn.close()
    
    print("✅ Fresh database created successfully!")

def test_database():
    """Test database connectivity and basic operations"""
    
    print("\n🧪 TESTING DATABASE")
    print("=" * 30)
    
    try:
        conn = sqlite3.connect('stock_app.db')
        cursor = conn.cursor()
        
        # Test user query
        cursor.execute("SELECT id, username FROM user LIMIT 1")
        user = cursor.fetchone()
        
        if user:
            print(f"✅ Database test passed")
            print(f"   Sample user: ID={user[0]}, Username={user[1]}")
        else:
            print("⚠️  No users found in database")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")

if __name__ == "__main__":
    repair_database()
    test_database()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Run: python app.py")
    print("2. Visit: http://localhost:5000")
    print("3. Login with: username=demo, password=demo123")
