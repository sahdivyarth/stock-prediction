"""
SQLite Command Line Helper
Provides step-by-step guidance for accessing the database
"""

import os
import subprocess
import sys
from pathlib import Path

def find_database():
    """Find the database file"""
    possible_paths = [
        'stock_app.db',
        'instance/stock_app.db',
        '../stock_app.db',
        './instance/stock_app.db'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def check_sqlite_installation():
    """Check if SQLite3 is installed"""
    try:
        result = subprocess.run(['sqlite3', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()
    except FileNotFoundError:
        return False, None

def print_tutorial():
    """Print step-by-step tutorial"""
    print("🗄️ SQLite Command Line Access Tutorial")
    print("=" * 50)
    print()
    
    # Check SQLite installation
    is_installed, version = check_sqlite_installation()
    
    if not is_installed:
        print("❌ SQLite3 is not installed or not in PATH")
        print()
        print("📥 How to install SQLite3:")
        print("   Windows: Download from https://sqlite.org/download.html")
        print("   Mac:     brew install sqlite3")
        print("   Linux:   sudo apt-get install sqlite3")
        print("   Or:      sudo yum install sqlite3")
        print()
        return
    
    print(f"✅ SQLite3 is installed: {version}")
    print()
    
    # Find database
    db_path = find_database()
    
    if not db_path:
        print("❌ Database file not found!")
        print()
        print("🔍 Looking for database in these locations:")
        print("   - stock_app.db")
        print("   - instance/stock_app.db")
        print("   - ../stock_app.db")
        print()
        print("💡 Make sure you're in the correct directory")
        print("   or run your Flask app first to create the database")
        return
    
    print(f"✅ Database found: {db_path}")
    print()
    
    # Show commands
    print("🚀 STEP-BY-STEP INSTRUCTIONS:")
    print("-" * 30)
    print()
    
    print("1️⃣ Open SQLite command line:")
    print(f"   sqlite3 {db_path}")
    print()
    
    print("2️⃣ Once inside SQLite (you'll see 'sqlite>' prompt), try these commands:")
    print()
    
    print("📋 Basic Commands:")
    print("   .tables              # Show all tables")
    print("   .schema              # Show table structures")
    print("   .headers on          # Show column headers")
    print("   .mode table          # Nice table format")
    print("   .help                # Get help")
    print("   .quit                # Exit SQLite")
    print()
    
    print("👥 View Users:")
    print("   SELECT * FROM user;")
    print()
    
    print("🔮 View Predictions:")
    print("   SELECT * FROM prediction_history;")
    print()
    
    print("📊 Join Query (Users + Predictions):")
    print("   SELECT u.username, p.ticker, p.model_used, p.created_at")
    print("   FROM prediction_history p")
    print("   JOIN user u ON p.user_id = u.id")
    print("   ORDER BY p.created_at DESC")
    print("   LIMIT 10;")
    print()
    
    print("📈 Statistics:")
    print("   SELECT COUNT(*) FROM user;                    # Total users")
    print("   SELECT COUNT(*) FROM prediction_history;     # Total predictions")
    print()
    
    print("💾 Export to CSV:")
    print("   .headers on")
    print("   .mode csv")
    print("   .output users.csv")
    print("   SELECT * FROM user;")
    print("   .output stdout")
    print("   .mode table")
    print()
    
    print("🔧 Database Maintenance:")
    print("   PRAGMA integrity_check;      # Check database health")
    print("   .dbinfo                      # Database information")
    print("   VACUUM;                      # Optimize database")
    print()
    
    print("🎯 QUICK START:")
    print("-" * 15)
    print(f"sqlite3 {db_path}")
    print(".tables")
    print("SELECT * FROM user;")
    print(".quit")
    print()

def interactive_sqlite_launcher():
    """Interactive launcher for SQLite"""
    print_tutorial()
    
    db_path = find_database()
    if not db_path:
        return
    
    print("🚀 Would you like to launch SQLite now? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes']:
        print(f"\n🔄 Launching SQLite with {db_path}...")
        print("💡 Type '.quit' to exit when you're done")
        print()
        
        try:
            # Launch SQLite
            subprocess.run(['sqlite3', db_path])
        except KeyboardInterrupt:
            print("\n👋 SQLite session ended")
        except Exception as e:
            print(f"❌ Error launching SQLite: {e}")

if __name__ == "__main__":
    interactive_sqlite_launcher()
