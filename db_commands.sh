#!/bin/bash

# Database access commands for the stock prediction app

echo "🗄️ STOCK PREDICTION APP - DATABASE ACCESS GUIDE"
echo "================================================"

echo ""
echo "📋 QUICK COMMANDS:"
echo ""

echo "1. 🔍 View database with command line:"
echo "   sqlite3 stock_app.db"
echo ""

echo "2. 🐍 Interactive Python manager:"
echo "   python database_manager.py"
echo ""

echo "3. 👀 Quick overview:"
echo "   python quick_db_viewer.py"
echo ""

echo "4. 🖥️ GUI browser:"
echo "   python db_browser_gui.py"
echo ""

echo "📊 COMMON SQL QUERIES:"
echo ""

echo "View all users:"
echo "   SELECT * FROM user;"
echo ""

echo "View recent predictions:"
echo "   SELECT p.*, u.username FROM prediction_history p JOIN user u ON p.user_id = u.id ORDER BY p.created_at DESC LIMIT 10;"
echo ""

echo "Count predictions by model:"
echo "   SELECT model_used, COUNT(*) FROM prediction_history GROUP BY model_used;"
echo ""

echo "User statistics:"
echo "   SELECT u.username, COUNT(p.id) as predictions FROM user u LEFT JOIN prediction_history p ON u.id = p.user_id GROUP BY u.username;"
echo ""

echo "🔧 MAINTENANCE COMMANDS:"
echo ""

echo "Backup database:"
echo "   cp stock_app.db stock_app_backup_$(date +%Y%m%d).db"
echo ""

echo "Export to CSV:"
echo "   sqlite3 -header -csv stock_app.db 'SELECT * FROM user;' > users.csv"
echo ""

echo "Check database integrity:"
echo "   sqlite3 stock_app.db 'PRAGMA integrity_check;'"
