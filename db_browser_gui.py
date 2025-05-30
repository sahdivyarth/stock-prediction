"""
Simple GUI database browser using tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import pandas as pd
from datetime import datetime

class DatabaseBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction App - Database Browser")
        self.root.geometry("1000x600")
        
        self.db_path = 'stock_app.db'
        
        self.create_widgets()
        self.load_data()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Users tab
        self.users_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.users_frame, text="Users")
        
        # Predictions tab
        self.predictions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predictions_frame, text="Predictions")
        
        # Create treeviews
        self.create_users_tab()
        self.create_predictions_tab()
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(buttons_frame, text="Refresh", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Export Users", command=self.export_users).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Export Predictions", command=self.export_predictions).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Backup DB", command=self.backup_database).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def create_users_tab(self):
        # Users treeview
        columns = ('ID', 'Username', 'Email', 'Created At', 'Predictions')
        self.users_tree = ttk.Treeview(self.users_frame, columns=columns, show='headings')
        
        for col in columns:
            self.users_tree.heading(col, text=col)
            self.users_tree.column(col, width=150)
        
        # Scrollbars
        users_scrollbar_y = ttk.Scrollbar(self.users_frame, orient=tk.VERTICAL, command=self.users_tree.yview)
        users_scrollbar_x = ttk.Scrollbar(self.users_frame, orient=tk.HORIZONTAL, command=self.users_tree.xview)
        self.users_tree.configure(yscrollcommand=users_scrollbar_y.set, xscrollcommand=users_scrollbar_x.set)
        
        # Grid layout
        self.users_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        users_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        users_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.users_frame.columnconfigure(0, weight=1)
        self.users_frame.rowconfigure(0, weight=1)
    
    def create_predictions_tab(self):
        # Predictions treeview
        columns = ('ID', 'User', 'Ticker', 'Model', 'Days', 'Created At')
        self.predictions_tree = ttk.Treeview(self.predictions_frame, columns=columns, show='headings')
        
        for col in columns:
            self.predictions_tree.heading(col, text=col)
            self.predictions_tree.column(col, width=120)
        
        # Scrollbars
        pred_scrollbar_y = ttk.Scrollbar(self.predictions_frame, orient=tk.VERTICAL, command=self.predictions_tree.yview)
        pred_scrollbar_x = ttk.Scrollbar(self.predictions_frame, orient=tk.HORIZONTAL, command=self.predictions_tree.xview)
        self.predictions_tree.configure(yscrollcommand=pred_scrollbar_y.set, xscrollcommand=pred_scrollbar_x.set)
        
        # Grid layout
        self.predictions_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pred_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        pred_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.predictions_frame.columnconfigure(0, weight=1)
        self.predictions_frame.rowconfigure(0, weight=1)
        
        # Double-click to view details
        self.predictions_tree.bind('<Double-1>', self.view_prediction_details)
    
    def load_data(self):
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load users
            users_query = """
                SELECT 
                    u.id,
                    u.username,
                    u.email,
                    u.created_at,
                    COUNT(p.id) as prediction_count
                FROM user u
                LEFT JOIN prediction_history p ON u.id = p.user_id
                GROUP BY u.id, u.username, u.email, u.created_at
                ORDER BY u.created_at DESC
            """
            
            users_df = pd.read_sql_query(users_query, conn)
            
            # Clear existing data
            for item in self.users_tree.get_children():
                self.users_tree.delete(item)
            
            # Insert users data
            for _, row in users_df.iterrows():
                self.users_tree.insert('', 'end', values=(
                    row['id'],
                    row['username'],
                    row['email'],
                    row['created_at'],
                    row['prediction_count']
                ))
            
            # Load predictions
            predictions_query = """
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
                LIMIT 100
            """
            
            predictions_df = pd.read_sql_query(predictions_query, conn)
            
            # Clear existing data
            for item in self.predictions_tree.get_children():
                self.predictions_tree.delete(item)
            
            # Insert predictions data
            for _, row in predictions_df.iterrows():
                self.predictions_tree.insert('', 'end', values=(
                    row['id'],
                    row['username'],
                    row['ticker'],
                    row['model_used'],
                    row['days_predicted'],
                    row['created_at']
                ))
            
            conn.close()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def view_prediction_details(self, event):
        selection = self.predictions_tree.selection()
        if selection:
            item = self.predictions_tree.item(selection[0])
            prediction_id = item['values'][0]
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT prediction_data FROM prediction_history WHERE id = ?
                """, (prediction_id,))
                
                result = cursor.fetchone()
                if result:
                    import json
                    data = json.loads(result[0])
                    
                    details = f"""
Prediction Details (ID: {prediction_id})

Current Price: ${data.get('current_price', 0):.2f}
Final Prediction: ${data.get('predictions', [0])[-1]:.2f}
Data Source: {data.get('data_source', 'Unknown')}

Model Metrics:
- MSE: {data.get('model_metrics', {}).get('mse', 0):.4f}
- MAE: {data.get('model_metrics', {}).get('mae', 0):.4f}
- Model: {data.get('model_metrics', {}).get('model_name', 'Unknown')}
                    """
                    
                    messagebox.showinfo("Prediction Details", details)
                
                conn.close()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load prediction details: {e}")
    
    def export_users(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query("SELECT * FROM user", conn)
                df.to_csv(filename, index=False)
                conn.close()
                messagebox.showinfo("Success", f"Users exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export users: {e}")
    
    def export_predictions(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query("""
                    SELECT 
                        p.*,
                        u.username
                    FROM prediction_history p
                    JOIN user u ON p.user_id = u.id
                """, conn)
                df.to_csv(filename, index=False)
                conn.close()
                messagebox.showinfo("Success", f"Predictions exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export predictions: {e}")
    
    def backup_database(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".db",
                filetypes=[("Database files", "*.db"), ("All files", "*.*")]
            )
            
            if filename:
                import shutil
                shutil.copy2(self.db_path, filename)
                messagebox.showinfo("Success", f"Database backed up to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to backup database: {e}")

def launch_gui():
    root = tk.Tk()
    app = DatabaseBrowser(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
