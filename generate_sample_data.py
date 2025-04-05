import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample Bitcoin price data
def generate_sample_data(days=100, volatility=0.02, initial_price=30000):
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate price data with random walk
    prices = [initial_price]
    for i in range(1, days):
        change = np.random.normal(0, volatility) 
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create dataframe
    df = pd.DataFrame({
        'datetime': dates,
        'Open': prices,
        'Close': prices,
        'timestamp': [int(date.timestamp()) for date in dates]
    })
    
    # Add other price and volume data
    df['High'] = df['Open'] * (1 + np.random.uniform(0, 0.03, size=len(df)))
    df['Low'] = df['Open'] * (1 - np.random.uniform(0, 0.02, size=len(df)))
    df['Volume'] = np.random.uniform(500, 1500, size=len(df))
    
    # Ensure proper order
    df = df[['timestamp', 'datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

# Generate and save the sample data
df = generate_sample_data(days=100)
df.to_csv('btcusd_1-min_data.csv', index=False)
print(f"Sample data generated with {len(df)} rows") 