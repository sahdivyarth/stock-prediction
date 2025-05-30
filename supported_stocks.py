"""
List of supported stocks and exchanges
"""

def get_supported_stocks():
    """Return comprehensive list of supported stocks"""
    
    stocks_by_category = {
        "Technology": [
            ("AAPL", "Apple Inc."),
            ("GOOGL", "Alphabet Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("AMZN", "Amazon.com Inc."),
            ("META", "Meta Platforms Inc."),
            ("NVDA", "NVIDIA Corporation"),
            ("TSLA", "Tesla Inc."),
            ("NFLX", "Netflix Inc."),
            ("ADBE", "Adobe Inc."),
            ("CRM", "Salesforce Inc."),
            ("ORCL", "Oracle Corporation"),
            ("INTC", "Intel Corporation"),
            ("AMD", "Advanced Micro Devices"),
            ("PYPL", "PayPal Holdings"),
            ("UBER", "Uber Technologies"),
            ("LYFT", "Lyft Inc."),
            ("SPOT", "Spotify Technology"),
            ("TWTR", "Twitter Inc."),
            ("SNAP", "Snap Inc."),
            ("ZM", "Zoom Video Communications"),
            ("DOCU", "DocuSign Inc."),
            ("SHOP", "Shopify Inc."),
            ("SQ", "Block Inc."),
            ("ROKU", "Roku Inc.")
        ],
        
        "Finance": [
            ("JPM", "JPMorgan Chase & Co."),
            ("BAC", "Bank of America Corp."),
            ("WFC", "Wells Fargo & Company"),
            ("GS", "Goldman Sachs Group"),
            ("MS", "Morgan Stanley"),
            ("C", "Citigroup Inc."),
            ("AXP", "American Express Company"),
            ("V", "Visa Inc."),
            ("MA", "Mastercard Inc."),
            ("BRK-B", "Berkshire Hathaway Inc.")
        ],
        
        "Healthcare": [
            ("JNJ", "Johnson & Johnson"),
            ("PFE", "Pfizer Inc."),
            ("UNH", "UnitedHealth Group"),
            ("MRNA", "Moderna Inc."),
            ("ABBV", "AbbVie Inc."),
            ("TMO", "Thermo Fisher Scientific"),
            ("DHR", "Danaher Corporation"),
            ("BMY", "Bristol Myers Squibb"),
            ("AMGN", "Amgen Inc."),
            ("GILD", "Gilead Sciences")
        ],
        
        "Consumer": [
            ("KO", "Coca-Cola Company"),
            ("PEP", "PepsiCo Inc."),
            ("WMT", "Walmart Inc."),
            ("HD", "Home Depot Inc."),
            ("MCD", "McDonald's Corporation"),
            ("SBUX", "Starbucks Corporation"),
            ("NKE", "Nike Inc."),
            ("DIS", "Walt Disney Company"),
            ("COST", "Costco Wholesale Corp."),
            ("TGT", "Target Corporation")
        ],
        
        "Energy": [
            ("XOM", "Exxon Mobil Corporation"),
            ("CVX", "Chevron Corporation"),
            ("COP", "ConocoPhillips"),
            ("EOG", "EOG Resources Inc."),
            ("SLB", "Schlumberger Limited"),
            ("PSX", "Phillips 66"),
            ("VLO", "Valero Energy Corporation"),
            ("MPC", "Marathon Petroleum Corp."),
            ("OXY", "Occidental Petroleum"),
            ("HAL", "Halliburton Company")
        ],
        
        "International": [
            ("BABA", "Alibaba Group (China)"),
            ("TSM", "Taiwan Semiconductor"),
            ("ASML", "ASML Holding (Netherlands)"),
            ("SAP", "SAP SE (Germany)"),
            ("TM", "Toyota Motor Corp. (Japan)"),
            ("NVO", "Novo Nordisk (Denmark)"),
            ("NESN.SW", "Nestlé (Switzerland)"),
            ("RHHBY", "Roche Holding (Switzerland)"),
            ("UL", "Unilever (UK/Netherlands)"),
            ("BP", "BP plc (UK)")
        ]
    }
    
    return stocks_by_category

def print_supported_stocks():
    """Print all supported stocks by category"""
    
    print("🌍 COMPREHENSIVE STOCK SUPPORT")
    print("=" * 50)
    
    stocks = get_supported_stocks()
    total_count = 0
    
    for category, stock_list in stocks.items():
        print(f"\n📊 {category.upper()} SECTOR:")
        print("-" * 30)
        
        for ticker, name in stock_list:
            print(f"   {ticker:<8} - {name}")
            total_count += 1
    
    print(f"\n📈 TOTAL FEATURED STOCKS: {total_count}")
    print("\n🔍 ADDITIONAL SUPPORT:")
    print("   • Any stock listed on major exchanges")
    print("   • NYSE, NASDAQ, LSE, TSE, and more")
    print("   • ETFs and Index Funds")
    print("   • International markets")
    print("   • Cryptocurrency (BTC-USD, ETH-USD, etc.)")
    
    print("\n💡 HOW TO USE:")
    print("   1. Enter any valid ticker symbol")
    print("   2. App automatically fetches real-time data")
    print("   3. Choose your preferred ML model")
    print("   4. Get instant predictions with charts")

if __name__ == "__main__":
    print_supported_stocks()
