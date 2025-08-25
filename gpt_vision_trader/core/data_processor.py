#!/usr/bin/env python3
"""
Data Processing Core Module
===========================

This module handles OHLCV data processing, window management, and preparation
for chart generation and GPT analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


class DataProcessor:
    """
    Processes OHLCV data for chart generation and analysis.
    Handles window extraction, technical indicator preparation, and data formatting.
    """
    
    def __init__(self, visible_days: int = 6, hidden_days: int = 1, indicator_buffer_days: int = 2):
        """
        Initialize data processor.
        
        Args:
            visible_days: Number of visible days in analysis window
            hidden_days: Number of hidden days (prediction horizon)
            indicator_buffer_days: Additional days for indicator pre-calculation
        """
        self.visible_days = visible_days
        self.hidden_days = hidden_days
        self.indicator_buffer_days = indicator_buffer_days
        
        logging.info(f"DataProcessor initialized: {visible_days} visible, {hidden_days} hidden, {indicator_buffer_days} buffer days")
    
    def calculate_required_candles(self, timeframe: str) -> int:
        """
        Calculate total number of candles needed including buffer for indicators.
        
        Args:
            timeframe: Timeframe string (e.g., '15m', '1h')
            
        Returns:
            Total number of candles to fetch
        """
        total_days = self.visible_days + self.indicator_buffer_days
        return self._calculate_candle_count(total_days, timeframe)
    
    def extract_analysis_window(self, 
                               ohlcv_data: pd.DataFrame,
                               timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract visible and hidden data windows for analysis.
        Uses full dataset (including buffer) for indicator calculation,
        but returns only visible portion for charting.
        
        Args:
            ohlcv_data: Full OHLCV dataset (including buffer days)
            timeframe: Timeframe string (e.g., '15m', '1h')
            
        Returns:
            Tuple of (visible_data_with_indicators, hidden_placeholder)
        """
        if ohlcv_data.empty:
            raise ValueError("OHLCV data is empty")
        
        # Calculate required candle counts
        visible_candles = self._calculate_candle_count(self.visible_days, timeframe)
        hidden_candles = self._calculate_candle_count(self.hidden_days, timeframe)
        
        # Use full dataset for indicator calculations (including buffer)
        # This ensures indicators have enough historical data for accurate calculation
        full_data_for_indicators = ohlcv_data.copy()
        
        # Extract only the visible portion for charting (most recent visible_candles)
        visible_data = full_data_for_indicators.tail(visible_candles).copy()
        
        # Create hidden placeholder with future timestamps
        if hidden_candles > 0:
            last_timestamp = visible_data.index[-1]
            future_timestamps = self._generate_future_timestamps(
                last_timestamp, hidden_candles, timeframe
            )
            
            hidden_placeholder = pd.DataFrame(
                index=future_timestamps,
                columns=visible_data.columns
            )
            hidden_placeholder[:] = float('nan')  # Fill with NaN for gaps
        else:
            hidden_placeholder = pd.DataFrame()
        
        logging.info(f"Extracted analysis window: {len(visible_data)} visible + {len(hidden_placeholder)} hidden (from {len(ohlcv_data)} total candles)")
        return visible_data, hidden_placeholder
    
    def prepare_metadata(self,
                        visible_data: pd.DataFrame,
                        chart_path: str,
                        pair: str,
                        timeframe: str,
                        additional_data: Optional[Dict] = None) -> Dict:
        """
        Prepare metadata for GPT analysis in backtest-compatible format.
        
        Args:
            visible_data: Visible OHLCV data
            chart_path: Path to generated chart
            pair: Trading pair
            timeframe: Timeframe
            additional_data: Additional metadata fields
            
        Returns:
            Metadata dictionary
        """
        import os
        
        chart_filename = os.path.basename(chart_path)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Format candle data
        visible_candles = self._format_candles_for_metadata(visible_data)
        
        # Create base metadata
        metadata = {
            chart_filename: {
                "result_file": chart_filename,
                "start_date": visible_data.index[0].strftime("%Y-%m-%d"),
                "visible_end_date": visible_data.index[-1].strftime("%Y-%m-%d"),
                "end_date": (visible_data.index[-1] + timedelta(days=self.hidden_days)).strftime("%Y-%m-%d"),
                "timestamp": timestamp,
                "pair": pair,
                "timeframe": timeframe,
                "symbol": pair,  # For backward compatibility
                "visible_candles": visible_candles,
                "hidden_candles": [],  # Empty for live trading
                "live_trading": True,
                "data_source": "live_processor"
            }
        }
        
        # Add additional data if provided
        if additional_data:
            metadata[chart_filename].update(additional_data)
        
        return metadata
    
    def calculate_performance(self, 
                            visible_data: pd.DataFrame,
                            hidden_data: pd.DataFrame) -> float:
        """
        Calculate performance between last visible and last hidden candle.
        
        Args:
            visible_data: Visible OHLCV data
            hidden_data: Hidden OHLCV data
            
        Returns:
            Performance as decimal (e.g., 0.05 for 5% gain)
        """
        if visible_data.empty or hidden_data.empty:
            return 0.0
        
        visible_close = visible_data['close'].iloc[-1]
        hidden_close = hidden_data['close'].iloc[-1]
        
        performance = (hidden_close / visible_close) - 1
        return round(performance, 4)
    
    def calculate_trend_classification(self,
                                     visible_data: pd.DataFrame,
                                     hidden_data: pd.DataFrame,
                                     atr_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate trend classification based on ATR.
        
        Args:
            visible_data: Visible OHLCV data
            hidden_data: Hidden OHLCV data
            atr_threshold: ATR multiplier threshold for trend classification
            
        Returns:
            Dictionary with trend information
        """
        if visible_data.empty or hidden_data.empty:
            return {'atr_factor': 0.0, 'trend': 'neutral', 'atr': 0.0}
        
        # Calculate ATR for visible data
        atr_period = 14
        atr_value = self._calculate_atr(visible_data, atr_period)
        
        # Calculate price change
        visible_close = visible_data['close'].iloc[-1]
        hidden_close = hidden_data['close'].iloc[-1]
        price_change = hidden_close - visible_close
        
        # Calculate ATR factor
        atr_factor = price_change / atr_value if atr_value > 0 else 0
        
        # Classify trend
        if atr_factor > atr_threshold:
            trend = 'bullish'
        elif atr_factor < -atr_threshold:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'atr_factor': round(atr_factor, 4),
            'trend': trend,
            'atr': round(atr_value, 4)
        }
    
    def _calculate_candle_count(self, days: int, timeframe: str) -> int:
        """
        Calculate number of candles for given days and timeframe.
        
        Args:
            days: Number of days
            timeframe: Timeframe string
            
        Returns:
            Number of candles
        """
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        candles_per_day = 1440 / timeframe_minutes  # 1440 minutes per day
        return int(days * candles_per_day)
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '15m', '1h', '4h', '1d')
            
        Returns:
            Number of minutes
        """
        timeframe = timeframe.lower()
        
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
    
    def _generate_future_timestamps(self,
                                  last_timestamp: pd.Timestamp,
                                  count: int,
                                  timeframe: str) -> pd.DatetimeIndex:
        """
        Generate future timestamps for hidden data placeholder.
        
        Args:
            last_timestamp: Last timestamp from visible data
            count: Number of future timestamps
            timeframe: Timeframe string
            
        Returns:
            DatetimeIndex with future timestamps
        """
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        
        future_timestamps = []
        current_timestamp = last_timestamp
        
        for _ in range(count):
            current_timestamp += timedelta(minutes=timeframe_minutes)
            future_timestamps.append(current_timestamp)
        
        return pd.DatetimeIndex(future_timestamps)
    
    def _format_candles_for_metadata(self, df: pd.DataFrame) -> List[Dict]:
        """
        Format OHLCV data for metadata storage.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of candle dictionaries
        """
        candles = []
        for timestamp, row in df.iterrows():
            candle = {
                "time": int(timestamp.timestamp() * 1000),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            }
            candles.append(candle)
        
        return candles
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range.
        
        Args:
            df: OHLCV DataFrame
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(df) < period:
            return 0.0
        
        # Calculate true range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        return float(atr.iloc[-1]) if not atr.empty else 0.0
