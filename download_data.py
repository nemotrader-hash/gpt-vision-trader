#!/usr/bin/env python3
"""
BTC Historical Data Downloader for Backtesting
Downloads OHLCV data from various CCXT exchanges
"""

import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import argparse

def download_btc_data(exchange_name='binance', symbol='BTC/USDT', timeframe='1h', 
                      start_date=None, end_date=None, limit=1000):
    """
    Download BTC historical data from specified exchange
    
    Args:
        exchange_name (str): Name of the exchange (e.g., 'binance', 'coinbase', 'kraken')
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT', 'BTC/USD')
        timeframe (str): Timeframe for data (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        limit (int): Number of candles to fetch per request
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    
    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Use spot trading data
            }
        })
        
        print(f"Connecting to {exchange_name}...")
        
        # Load markets
        exchange.load_markets()
        
        if symbol not in exchange.markets:
            print(f"Symbol {symbol} not found on {exchange_name}")
            print(f"Available symbols: {list(exchange.markets.keys())[:10]}...")
            return None
        
        # Set default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Downloading {symbol} data from {start_date} to {end_date}")
        print(f"Timeframe: {timeframe}")
        
        # Convert dates to timestamps
        start_timestamp = exchange.parse8601(start_date + 'T00:00:00Z')
        end_timestamp = exchange.parse8601(end_date + 'T23:59:59Z')
        
        all_data = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=current_timestamp, 
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Update timestamp for next iteration
                current_timestamp = ohlcv[-1][0] + 1
                
                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)
                
                print(f"Downloaded {len(ohlcv)} candles, total: {len(all_data)}")
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_data:
            print("No data downloaded")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates and sort
        df = df.drop_duplicates().sort_index()
        
        # Filter by date range
        df = df[start_date:end_date]
        
        print(f"Downloaded {len(df)} data points")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"Error initializing {exchange_name}: {e}")
        return None

def save_data(df, exchange_name, symbol, timeframe, start_date, end_date):
    """Save data to CSV file"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create filename
    symbol_clean = symbol.replace('/', '_')
    filename = f"data/{exchange_name}_{symbol_clean}_{timeframe}_{start_date}_{end_date}.csv"
    
    # Save to CSV
    df.to_csv(filename)
    print(f"Data saved to: {filename}")
    
    return filename

def main():
    parser = argparse.ArgumentParser(description='Download BTC historical data for backtesting')
    parser.add_argument('--exchange', default='binance', help='Exchange name (default: binance)')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to download (default: 30)')
    
    args = parser.parse_args()
    
    # Set default dates based on days parameter
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=== BTC Historical Data Downloader ===")
    print(f"Exchange: {args.exchange}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print("=" * 40)
    
    # Download data
    df = download_btc_data(
        exchange_name=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if df is not None:
        # Save data
        filename = save_data(df, args.exchange, args.symbol, args.timeframe, 
                           args.start_date, args.end_date)
        
        # Display sample data
        print("\nSample data:")
        print(df.head())
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Basic statistics
        print("\nBasic statistics:")
        print(df.describe())
        
    else:
        print("Failed to download data")

if __name__ == "__main__":
    main()
