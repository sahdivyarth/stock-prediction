import pytz
from datetime import datetime

def test_timezone():
    print("🧪 TESTING TIMEZONE FUNCTIONALITY")
    print("=" * 50)
    
    # Get current time in different ways
    utc_now = datetime.utcnow()
    local_now = datetime.now()
    
    # IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = datetime.now(ist)
    
    print(f"UTC Time (utcnow): {utc_now}")
    print(f"Local Time (now): {local_now}")
    print(f"IST Time (now with IST): {ist_now}")
    print(f"IST Formatted: {ist_now.strftime('%d %B %Y at %I:%M %p IST')}")
    
    # Convert UTC to IST
    utc_localized = pytz.UTC.localize(utc_now)
    ist_converted = utc_localized.astimezone(ist)
    print(f"UTC to IST Converted: {ist_converted}")
    print(f"IST Converted Formatted: {ist_converted.strftime('%d %B %Y at %I:%M %p IST')}")
    
    # Test what we should store in database
    ist_naive = ist_now.replace(tzinfo=None)
    print(f"IST Naive (for database): {ist_naive}")
    
    # Test conversion back
    ist_localized = ist.localize(ist_naive)
    print(f"IST Localized back: {ist_localized}")
    
    print("\n✅ Timezone test completed!")

if __name__ == "__main__":
    test_timezone()
