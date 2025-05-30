@echo off
REM Windows batch file to access SQLite database

echo 🗄️ SQLite Database Access - Windows
echo ===================================
echo.

REM Check if SQLite3 is available
sqlite3 -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ SQLite3 is not installed or not in PATH
    echo.
    echo 📥 To install SQLite3 on Windows:
    echo    1. Download from https://sqlite.org/download.html
    echo    2. Extract sqlite3.exe to a folder
    echo    3. Add that folder to your PATH environment variable
    echo    4. Or place sqlite3.exe in your project folder
    echo.
    pause
    exit /b 1
)

echo ✅ SQLite3 is available
echo.

REM Find database file
if exist "stock_app.db" (
    set DB_PATH=stock_app.db
    echo ✅ Found database: stock_app.db
) else if exist "instance\stock_app.db" (
    set DB_PATH=instance\stock_app.db
    echo ✅ Found database: instance\stock_app.db
) else (
    echo ❌ Database file not found!
    echo    Looking for: stock_app.db or instance\stock_app.db
    echo    Make sure you're in the correct directory
    echo.
    pause
    exit /b 1
)

echo.
echo 🚀 Starting SQLite command line...
echo 💡 Type .quit to exit when done
echo.
echo 📋 Quick commands to try:
echo    .tables
echo    SELECT * FROM user;
echo    SELECT * FROM prediction_history;
echo    .quit
echo.
pause

REM Launch SQLite
sqlite3 %DB_PATH%

echo.
echo 👋 SQLite session ended
pause
