#!/bin/bash
# SQLite Command Line Tutorial for Stock Prediction App

echo "🗄️ SQLite Command Line Tutorial"
echo "================================"
echo ""
echo "This tutorial shows you how to access your stock_app.db database"
echo "using the SQLite command line interface."
echo ""

# Check if SQLite is installed
if command -v sqlite3 &> /dev/null; then
    echo "✅ SQLite3 is installed and ready to use"
else
    echo "❌ SQLite3 is not installed. Please install it first:"
    echo "   Windows: Download from https://sqlite.org/download.html"
    echo "   Mac: brew install sqlite3"
    echo "   Linux: sudo apt-get install sqlite3"
    exit 1
fi

echo ""
echo "📍 Current directory: $(pwd)"
echo "🔍 Looking for database files..."

# Find database files
if [ -f "stock_app.db" ]; then
    echo "✅ Found: stock_app.db"
elif [ -f "instance/stock_app.db" ]; then
    echo "✅ Found: instance/stock_app.db"
else
    echo "❌ Database file not found. Make sure you're in the correct directory."
    echo "   The database should be named 'stock_app.db' or 'instance/stock_app.db'"
fi

echo ""
echo "🚀 To start SQLite command line, run one of these commands:"
echo "   sqlite3 stock_app.db"
echo "   sqlite3 instance/stock_app.db"
echo ""
echo "📋 Once inside SQLite, you can use the commands shown below..."
