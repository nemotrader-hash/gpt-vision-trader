#!/usr/bin/env python3
"""
Freqtrade API Client with GPT Reasoning
=======================================

Freqtrade REST API client that uses the official ft_rest_client
and includes GPT reasoning responses in entry tags for better trade tracking.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .ft_rest_client import FtRestClient


@dataclass 
class FreqtradeConfig:
    """Configuration for Freqtrade REST API connection."""
    base_url: str = "http://127.0.0.1:8080"
    username: str = "freqtrade"
    password: str = "SuperSecret1!"
    timeout: int = 30


class FreqtradeAPIClient:
    """
    Freqtrade API client that uses the official ft_rest_client
    and includes GPT reasoning in entry tags.
    
    Key features:
    - Uses official Freqtrade REST client
    - Includes GPT reasoning in entry tags
    - Sanitizes reasoning text for valid tags
    - Maintains backward compatibility with existing code
    """
    
    def __init__(self, config: FreqtradeConfig):
        """
        Initialize the enhanced Freqtrade API client.
        
        Args:
            config: Freqtrade API configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the official Freqtrade REST client
        self.client = FtRestClient(
            serverurl=config.base_url,
            username=config.username,
            password=config.password,
            timeout=config.timeout
        )
        
        self.logger.info(f"FreqtradeAPIClient initialized for {config.base_url}")
    
    def _sanitize_entry_tag(self, reasoning: str, max_length: int = 100) -> str:
        """
        Sanitize GPT reasoning for use as entry tag.
        
        Args:
            reasoning: GPT reasoning text
            max_length: Maximum tag length
            
        Returns:
            Sanitized entry tag
        """
        if not reasoning:
            return "gpt_signal"
        
        # Remove special characters and keep only alphanumeric, spaces, and basic punctuation
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-_.,!?]', '', reasoning)
        
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length-3] + "..."
        
        # Replace spaces with underscores for tag format
        sanitized = sanitized.replace(' ', '_')
        
        # Ensure it starts with a letter (requirement for some systems)
        if sanitized and not sanitized[0].isalpha():
            sanitized = f"gpt_{sanitized}"
        
        return sanitized or "gpt_signal"
    
    def ping(self) -> bool:
        """
        Test API connection.
        
        Returns:
            True if API is accessible
        """
        try:
            response = self.client.ping()
            return response and response.get('status') == 'pong'
        except Exception as e:
            self.logger.error(f"Ping failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get bot status and open trades.
        
        Returns:
            Bot status information including open trades
        """
        try:
            response = self.client.status()
            
            if isinstance(response, list):
                # Handle case where status returns list of trades
                return {
                    'state': 'running',
                    'dry_run': True,
                    'max_open_trades': 3,
                    'stake_currency': 'USDT',
                    'open_trades': response
                }
            elif isinstance(response, dict):
                return response
            else:
                return {
                    'state': 'unknown',
                    'dry_run': True,
                    'max_open_trades': 0,
                    'stake_currency': 'USDT',
                    'open_trades': []
                }
        except Exception as e:
            self.logger.error(f"Failed to get status: {e}")
            return {
                'state': 'error',
                'dry_run': True,
                'max_open_trades': 0,
                'stake_currency': 'USDT',
                'open_trades': []
            }
    
    def get_pair_candles(self, 
                        pair: str, 
                        timeframe: str, 
                        limit: int = 500) -> pd.DataFrame:
        """
        Get OHLCV candle data for a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '15m', '1h', '4h')
            limit: Number of candles to retrieve (max 1500)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            response = self.client.pair_candles(
                pair=pair,
                timeframe=timeframe,
                limit=min(limit, 1500)
            )
            
            if not response:
                self.logger.warning(f"No response for {pair} {timeframe}")
                return pd.DataFrame()
            
            # Handle different response formats
            candle_data = None
            if isinstance(response, dict) and 'data' in response:
                candle_data = response['data']
            elif isinstance(response, list):
                candle_data = response
            
            if candle_data:
                # Check if data is list of lists (Freqtrade format) or list of dicts
                if isinstance(candle_data[0], list):
                    # List of lists format: [timestamp, open, high, low, close, volume, ...]
                    df_data = []
                    for candle in candle_data:
                        if len(candle) >= 6:  # Ensure we have at least OHLCV data
                            df_data.append({
                                'date': candle[0],
                                'open': candle[1],
                                'high': candle[2], 
                                'low': candle[3],
                                'close': candle[4],
                                'volume': candle[5]
                            })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                    else:
                        self.logger.error("No valid candle data found")
                        return pd.DataFrame()
                else:
                    # Dictionary format
                    df = pd.DataFrame(candle_data)
                
                if df.empty:
                    self.logger.warning(f"Empty DataFrame for {pair}")
                    return pd.DataFrame()
                
                # Convert timestamp and set as index
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # Convert to numeric types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                available_columns = [col for col in numeric_columns if col in df.columns]
                
                if available_columns:
                    df[available_columns] = df[available_columns].apply(pd.to_numeric, errors='coerce')
                
                self.logger.info(f"✅ Retrieved {len(df)} candles for {pair} {timeframe}")
                return df
            else:
                self.logger.warning(f"No candle data found for {pair} {timeframe}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to get candles for {pair} {timeframe}: {e}")
            return pd.DataFrame()
    
    def force_enter_with_reasoning(self, 
                                  pair: str, 
                                  side: str,
                                  gpt_reasoning: str,
                                  prediction: str = None,
                                  price: float = None) -> Dict[str, Any]:
        """
        Force entry into a position with GPT reasoning as entry tag.
        
        Args:
            pair: Trading pair
            side: 'long' or 'short'
            gpt_reasoning: GPT reasoning text to include in entry tag
            prediction: GPT prediction (bullish/bearish/neutral)
            price: Optional entry price
            
        Returns:
            Response from force entry command
        """
        try:
            # Create comprehensive entry tag with reasoning
            base_tag = f"gpt_{prediction or side}" if prediction else f"gpt_{side}"
            reasoning_tag = self._sanitize_entry_tag(gpt_reasoning, max_length=80)
            
            # Combine base tag with reasoning (keep within reasonable limits)
            if reasoning_tag and reasoning_tag != "gpt_signal":
                entry_tag = f"{base_tag}_{reasoning_tag}"
            else:
                entry_tag = base_tag
            
            # Ensure tag isn't too long (some systems have limits)
            if len(entry_tag) > 100:
                entry_tag = entry_tag[:97] + "..."
            
            self.logger.info(f"Entering {side} position for {pair}")
            self.logger.info(f"GPT Reasoning: {gpt_reasoning[:200]}...")
            self.logger.info(f"Entry tag: {entry_tag}")
            
            # Use the official Freqtrade client
            response = self.client.forceenter(
                pair=pair,
                side=side,
                enter_tag=entry_tag,
                price=price
            )
            
            if response:
                self.logger.info(f"✅ Force entry executed for {pair} {side}")
                self.logger.info(f"Response: {response}")
            else:
                self.logger.warning(f"⚠️ Force entry may have failed for {pair}")
            
            return response or {}
            
        except Exception as e:
            self.logger.error(f"Failed to force entry for {pair} {side}: {e}")
            raise
    
    def force_exit(self, 
                  trade_id: int,
                  order_type: str = None,
                  amount: float = None) -> Dict[str, Any]:
        """
        Force exit from a position.
        
        Args:
            trade_id: Trade ID to exit
            order_type: Order type (market/limit)
            amount: Amount to exit (None for full exit)
            
        Returns:
            Response from force exit command
        """
        try:
            response = self.client.forceexit(
                tradeid=trade_id,
                ordertype=order_type,
                amount=amount
            )
            
            self.logger.info(f"✅ Force exit executed for trade {trade_id}")
            return response or {}
            
        except Exception as e:
            self.logger.error(f"Failed to force exit trade {trade_id}: {e}")
            raise
    
    def get_open_trades(self) -> List[Dict[str, Any]]:
        """
        Get list of currently open trades.
        
        Returns:
            List of open trade dictionaries
        """
        try:
            status = self.get_status()
            return status.get('open_trades', [])
        except Exception as e:
            self.logger.error(f"Failed to get open trades: {e}")
            return []
    
    def get_profit_info(self) -> Dict[str, Any]:
        """
        Get profit information and statistics.
        
        Returns:
            Profit information dictionary
        """
        try:
            return self.client.profit() or {}
        except Exception as e:
            self.logger.error(f"Failed to get profit info: {e}")
            return {}
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive trading summary.
        
        Returns:
            Dictionary with trading summary information
        """
        try:
            status = self.get_status()
            profit = self.get_profit_info()
            open_trades = self.get_open_trades()
            
            summary = {
                'bot_state': status.get('state', 'unknown'),
                'dry_run': status.get('dry_run', True),
                'open_trades_count': len(open_trades),
                'max_open_trades': status.get('max_open_trades', 0),
                'stake_currency': status.get('stake_currency', 'USDT'),
                'total_profit': profit.get('profit_total', 0),
                'profit_ratio': profit.get('profit_ratio', 0),
                'trade_count': profit.get('trade_count', 0),
                'win_rate': profit.get('winrate', 0),
                'open_trades': open_trades
            }
            
            self.logger.info(f"Trading summary: {summary['bot_state']}, {summary['open_trades_count']} open trades")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get trading summary: {e}")
            return {
                'bot_state': 'error',
                'dry_run': True,
                'open_trades_count': 0,
                'max_open_trades': 0,
                'stake_currency': 'USDT',
                'total_profit': 0,
                'profit_ratio': 0,
                'trade_count': 0,
                'win_rate': 0,
                'open_trades': [],
                'error': str(e)
            }
    
    def is_dry_run(self) -> bool:
        """
        Check if bot is running in dry run mode.
        
        Returns:
            True if in dry run mode
        """
        try:
            status = self.get_status()
            return status.get('dry_run', True)
        except Exception:
            return True


class TradingOperations:
    """
    Trading operations that include GPT reasoning in trades.
    """
    
    def __init__(self, api_client: FreqtradeAPIClient):
        """
        Initialize trading operations.
        
        Args:
            api_client: Freqtrade API client instance
        """
        if not isinstance(api_client, FreqtradeAPIClient):
            raise TypeError(f"Expected FreqtradeAPIClient, got {type(api_client)}")
        self.api = api_client
        self.logger = logging.getLogger(__name__)
    
    def enter_position_with_reasoning(self, 
                                    pair: str, 
                                    direction: str, 
                                    gpt_analysis: Dict[str, Any]) -> bool:
        """
        Enter a position with GPT reasoning included in entry tag.
        
        Args:
            pair: Trading pair
            direction: 'long' or 'short'
            gpt_analysis: GPT analysis result containing reasoning
            
        Returns:
            True if successful
        """
        try:
            # Check if position already exists
            open_trades = self.api.get_open_trades()
            existing_trade = next((t for t in open_trades if t.get('pair') == pair), None)
            
            if existing_trade:
                self.logger.info(f"Position already exists for {pair}, skipping entry")
                return False
            
            # Extract reasoning and prediction from GPT analysis
            reasoning = gpt_analysis.get('analysis', 'GPT analysis')
            prediction = gpt_analysis.get('prediction', direction)
            
            # Execute force entry with reasoning
            response = self.api.force_enter_with_reasoning(
                pair=pair,
                side=direction,
                gpt_reasoning=reasoning,
                prediction=prediction
            )
            
            success = bool(response)
            if success:
                self.logger.info(f"✅ Successfully entered {direction} position for {pair} with GPT reasoning")
            else:
                self.logger.warning(f"⚠️ Force entry may have failed for {pair}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error entering position for {pair}: {e}")
            return False
    
    def exit_position(self, pair: str, reason: str = "gpt_exit") -> bool:
        """
        Exit position for a trading pair.
        
        Args:
            pair: Trading pair
            reason: Reason for exit
            
        Returns:
            True if successful
        """
        try:
            # Find open trade for this pair
            open_trades = self.api.get_open_trades()
            existing_trade = next((t for t in open_trades if t.get('pair') == pair), None)
            
            if not existing_trade:
                self.logger.info(f"No position exists for {pair}, skipping exit")
                return False
            
            trade_id = existing_trade.get('trade_id')
            if not trade_id:
                self.logger.error(f"No trade ID found for {pair}")
                return False
            
            # Execute force exit
            response = self.api.force_exit(trade_id=trade_id)
            
            success = bool(response)
            if success:
                self.logger.info(f"✅ Successfully exited position for {pair}")
            else:
                self.logger.warning(f"⚠️ Force exit may have failed for {pair}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error exiting position for {pair}: {e}")
            return False
    
    def execute_trading_signals_with_reasoning(self, 
                                             signals: Dict[str, bool], 
                                             pair: str,
                                             gpt_analysis: Dict[str, Any]) -> Dict[str, bool]:
        """
        Execute trading signals with GPT reasoning included.
        
        Args:
            signals: Trading signals dictionary
            pair: Trading pair
            gpt_analysis: GPT analysis result
            
        Returns:
            Dictionary with execution results
        """
        results = {
            'enter_long_success': False,
            'exit_long_success': False,
            'enter_short_success': False,
            'exit_short_success': False
        }
        
        try:
            # Process exit signals first
            if signals.get('exit_long') or signals.get('exit_short'):
                results['exit_long_success'] = self.exit_position(pair, "gpt_exit")
                results['exit_short_success'] = results['exit_long_success']  # Same operation
            
            # Process entry signals with reasoning
            if signals.get('enter_long'):
                results['enter_long_success'] = self.enter_position_with_reasoning(
                    pair, 'long', gpt_analysis
                )
            elif signals.get('enter_short'):
                results['enter_short_success'] = self.enter_position_with_reasoning(
                    pair, 'short', gpt_analysis
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing trading signals for {pair}: {e}")
            return results
