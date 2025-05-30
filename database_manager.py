"""
Complete database management tool for the stock prediction app
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

class DatabaseManager:
    def __init__(self, db_path='stock_app.db'):
        self.db_path = db_path
        
    def connect(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)
    
    def view_all_tables(self):
        """Show all tables in the database"""
        print("📊 DATABASE TABLES")
        print("=" * 40)
        
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"📋 {table_name}: {count} records")
        
        conn.close()
        return [table[0] for table in tables]
    
    def view_users(self):
        """View all users in the database"""
        print("\n👥 USERS TABLE")
        print("=" * 50)
        
        conn = self.connect()
        
        try:
            df = pd.read_sql_query("""
                SELECT id, username, email, created_at 
                FROM user 
                ORDER BY created_at DESC
            """, conn)
            
            if df.empty:
                print("No users found")
            else:
                print(df.to_string(index=False))
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
    
    def view_predictions(self, limit=20):
        """View recent predictions"""
        print(f"\n🔮 RECENT PREDICTIONS (Last {limit})")
        print("=" * 60)
        
        conn = self.connect()
        
        try:
            df = pd.read_sql_query(f"""
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
                LIMIT {limit}
            """, conn)
            
            if df.empty:
                print("No predictions found")
            else:
                print(df.to_string(index=False))
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
    
    def view_prediction_details(self, prediction_id):
        """View detailed prediction data"""
        print(f"\n📈 PREDICTION DETAILS (ID: {prediction_id})")
        print("=" * 50)
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    p.ticker,
                    p.model_used,
                    p.days_predicted,
                    p.prediction_data,
                    p.created_at,
                    u.username
                FROM prediction_history p
                JOIN user u ON p.user_id = u.id
                WHERE p.id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            
            if result:
                ticker, model, days, data_json, created_at, username = result
                
                print(f"User: {username}")
                print(f"Ticker: {ticker}")
                print(f"Model: {model}")
                print(f"Days Predicted: {days}")
                print(f"Created: {created_at}")
                print("\nPrediction Data:")
                
                # Parse and display prediction data
                try:
                    data = json.loads(data_json)
                    print(f"Current Price: ${data.get('current_price', 0):.2f}")
                    
                    predictions = data.get('predictions', [])
                    if predictions:
                        print(f"Final Prediction: ${predictions[-1]:.2f}")
                        print(f"Predicted Change: {((predictions[-1] - data.get('current_price', 0)) / data.get('current_price', 1) * 100):+.2f}%")
                    
                    metrics = data.get('model_metrics', {})
                    print(f"Model MSE: {metrics.get('mse', 0):.4f}")
                    print(f"Model MAE: {metrics.get('mae', 0):.4f}")
                    
                except json.JSONDecodeError:
                    print("Could not parse prediction data")
            else:
                print("Prediction not found")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
    
    def search_predictions(self, ticker=None, username=None, model=None):
        """Search predictions by criteria"""
        print("\n🔍 SEARCH RESULTS")
        print("=" * 40)
        
        conn = self.connect()
        
        # Build query
        query = """
            SELECT 
                p.id,
                u.username,
                p.ticker,
                p.model_used,
                p.days_predicted,
                p.created_at
            FROM prediction_history p
            JOIN user u ON p.user_id = u.id
            WHERE 1=1
        """
        params = []
        
        if ticker:
            query += " AND p.ticker = ?"
            params.append(ticker.upper())
        
        if username:
            query += " AND u.username = ?"
            params.append(username)
        
        if model:
            query += " AND p.model_used = ?"
            params.append(model)
        
        query += " ORDER BY p.created_at DESC"
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                print("No predictions found matching criteria")
            else:
                print(df.to_string(index=False))
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
    
    def get_user_stats(self, username=None):
        """Get user statistics"""
        print("\n📊 USER STATISTICS")
        print("=" * 40)
        
        conn = self.connect()
        
        try:
            if username:
                # Stats for specific user
                query = """
                    SELECT 
                        u.username,
                        COUNT(p.id) as total_predictions,
                        COUNT(DISTINCT p.ticker) as unique_tickers,
                        p.model_used,
                        COUNT(*) as model_count
                    FROM user u
                    LEFT JOIN prediction_history p ON u.id = p.user_id
                    WHERE u.username = ?
                    GROUP BY u.username, p.model_used
                """
                df = pd.read_sql_query(query, conn, params=[username])
            else:
                # Stats for all users
                query = """
                    SELECT 
                        u.username,
                        COUNT(p.id) as total_predictions,
                        COUNT(DISTINCT p.ticker) as unique_tickers,
                        MAX(p.created_at) as last_prediction
                    FROM user u
                    LEFT JOIN prediction_history p ON u.id = p.user_id
                    GROUP BY u.username
                    ORDER BY total_predictions DESC
                """
                df = pd.read_sql_query(query, conn)
            
            if df.empty:
                print("No user data found")
            else:
                print(df.to_string(index=False))
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
    
    def create_user(self, username, email, password):
        """Create a new user"""
        print(f"\n➕ CREATING USER: {username}")
        print("=" * 30)
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Check if user exists
            cursor.execute("SELECT id FROM user WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                print("❌ User with this username or email already exists")
                return False
            
            # Create user
            hashed_password = generate_password_hash(password)
            cursor.execute("""
                INSERT INTO user (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, email, hashed_password, datetime.utcnow()))
            
            conn.commit()
            print(f"✅ User '{username}' created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def delete_user(self, username):
        """Delete a user and their predictions"""
        print(f"\n🗑️ DELETING USER: {username}")
        print("=" * 30)
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Get user ID
            cursor.execute("SELECT id FROM user WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            if not user:
                print("❌ User not found")
                return False
            
            user_id = user[0]
            
            # Delete predictions first
            cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
            predictions_deleted = cursor.rowcount
            
            # Delete user
            cursor.execute("DELETE FROM user WHERE id = ?", (user_id,))
            
            conn.commit()
            print(f"✅ User '{username}' deleted successfully")
            print(f"   Also deleted {predictions_deleted} predictions")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting user: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def export_data(self, table_name, filename=None):
        """Export table data to CSV"""
        if not filename:
            filename = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print(f"\n📤 EXPORTING {table_name.upper()} TO {filename}")
        print("=" * 50)
        
        conn = self.connect()
        
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(filename, index=False)
            print(f"✅ Exported {len(df)} records to {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
        finally:
            conn.close()
    
    def backup_database(self, backup_path=None):
        """Create a backup of the database"""
        if not backup_path:
            backup_path = f"stock_app_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        print(f"\n💾 CREATING BACKUP: {backup_path}")
        print("=" * 40)
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"✅ Database backed up to {backup_path}")
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
    
    def execute_custom_query(self, query):
        """Execute a custom SQL query"""
        print(f"\n🔧 EXECUTING CUSTOM QUERY")
        print("=" * 30)
        print(f"Query: {query}")
        print()
        
        conn = self.connect()
        
        try:
            if query.strip().upper().startswith('SELECT'):
                # Read query
                df = pd.read_sql_query(query, conn)
                if df.empty:
                    print("No results found")
                else:
                    print(df.to_string(index=False))
            else:
                # Write query
                cursor = conn.cursor()
                cursor.execute(query)
                conn.commit()
                print(f"✅ Query executed successfully. Rows affected: {cursor.rowcount}")
                
        except Exception as e:
            print(f"❌ Error executing query: {e}")
        finally:
            conn.close()

def interactive_database_manager():
    """Interactive database management interface"""
    
    db_manager = DatabaseManager()
    
    print("🗄️ STOCK PREDICTION APP - DATABASE MANAGER")
    print("=" * 50)
    
    while True:
        print("\n📋 AVAILABLE COMMANDS:")
        print("1.  View all tables")
        print("2.  View users")
        print("3.  View predictions")
        print("4.  View prediction details")
        print("5.  Search predictions")
        print("6.  User statistics")
        print("7.  Create user")
        print("8.  Delete user")
        print("9.  Export data")
        print("10. Backup database")
        print("11. Custom SQL query")
        print("12. Exit")
        
        choice = input("\nEnter your choice (1-12): ").strip()
        
        try:
            if choice == '1':
                db_manager.view_all_tables()
                
            elif choice == '2':
                db_manager.view_users()
                
            elif choice == '3':
                limit = input("Number of predictions to show (default 20): ").strip()
                limit = int(limit) if limit else 20
                db_manager.view_predictions(limit)
                
            elif choice == '4':
                pred_id = input("Enter prediction ID: ").strip()
                if pred_id:
                    db_manager.view_prediction_details(int(pred_id))
                    
            elif choice == '5':
                ticker = input("Ticker (optional): ").strip() or None
                username = input("Username (optional): ").strip() or None
                model = input("Model (optional): ").strip() or None
                db_manager.search_predictions(ticker, username, model)
                
            elif choice == '6':
                username = input("Username (optional, leave blank for all): ").strip() or None
                db_manager.get_user_stats(username)
                
            elif choice == '7':
                username = input("New username: ").strip()
                email = input("Email: ").strip()
                password = input("Password: ").strip()
                if username and email and password:
                    db_manager.create_user(username, email, password)
                    
            elif choice == '8':
                username = input("Username to delete: ").strip()
                if username:
                    confirm = input(f"Are you sure you want to delete '{username}'? (yes/no): ").strip()
                    if confirm.lower() == 'yes':
                        db_manager.delete_user(username)
                        
            elif choice == '9':
                table = input("Table name (user/prediction_history): ").strip()
                filename = input("Filename (optional): ").strip() or None
                if table:
                    db_manager.export_data(table, filename)
                    
            elif choice == '10':
                backup_path = input("Backup filename (optional): ").strip() or None
                db_manager.backup_database(backup_path)
                
            elif choice == '11':
                query = input("Enter SQL query: ").strip()
                if query:
                    db_manager.execute_custom_query(query)
                    
            elif choice == '12':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
        except ValueError:
            print("❌ Invalid input. Please try again.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_database_manager()
