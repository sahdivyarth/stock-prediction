"""
Quick database viewer for common operations
"""

import sqlite3
import pandas as pd
from datetime import datetime

def quick_view():
    """Quick view of database contents"""
    
    print("🔍 QUICK DATABASE OVERVIEW")
    print("=" * 40)
    
    try:
        conn = sqlite3.connect('stock_app.db')
        
        # Users
        print("\n👥 USERS:")
        users_df = pd.read_sql_query("SELECT id, username, email, created_at FROM user", conn)
        print(users_df.to_string(index=False))
        
        # Recent predictions
        print("\n🔮 RECENT PREDICTIONS:")
        predictions_df = pd.read_sql_query("""
            SELECT 
                p.id,
                u.username,
                p.ticker,
                p.model_used,
                p.created_at
            FROM prediction_history p
            JOIN user u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            LIMIT 10
        """, conn)
        
        if predictions_df.empty:
            print("No predictions found")
        else:
            print(predictions_df.to_string(index=False))
        
        # Statistics
        print("\n📊 STATISTICS:")
        stats = pd.read_sql_query("""
            SELECT 
                COUNT(DISTINCT u.id) as total_users,
                COUNT(p.id) as total_predictions,
                COUNT(DISTINCT p.ticker) as unique_tickers
            FROM user u
            LEFT JOIN prediction_history p ON u.id = p.user_id
        """, conn)
        
        print(f"Total Users: {stats.iloc[0]['total_users']}")
        print(f"Total Predictions: {stats.iloc[0]['total_predictions']}")
        print(f"Unique Tickers: {stats.iloc[0]['unique_tickers']}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_view()
