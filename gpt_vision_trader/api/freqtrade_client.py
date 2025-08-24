#!/usr/bin/env python3
"""
Freqtrade REST API Client
=========================

Simplified and improved Freqtrade REST API client focused on essential trading operations.
Removes stake amount calculations - lets Freqtrade handle position sizing automatically.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


@dataclass
class FreqtradeConfig:
    """Configuration for Freqtrade REST API connection."""
    base_url: str = "http://127.0.0.1:8080"
    username: str = "freqtrade"
    password: str = "SuperSecret1!"
    timeout: int = 30


class FreqtradeAPIClient:
    """
    Simplified Freqtrade REST API client focused on essential trading operations.
    
    Key simplifications:
    - No stake amount calculations (handled by Freqtrade)
    - Focus on forceenter/forceexit operations
    - Streamlined data retrieval methods
    """
    
    def __init__(self, config: FreqtradeConfig):
        """
        Initialize the Freqtrade API client.
        
        Args:
            config: Freqtrade API configuration
        """
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.session = requests.Session()
        self.session.auth = (config.username, config.password)
        self.session.timeout = config.timeout
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FreqtradeAPIClient initialized for {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an authenticated request to the Freqtrade API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without /api/v1 prefix)
            **kwargs: Additional arguments for requests
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/api/v1{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {method} {url} - {e}")
            raise
    
    def ping(self) -> bool:
        """
        Test API connection.
        
        Returns:
            True if API is accessible
        """
        try:
            response = self._make_request('GET', '/ping')
            return response.get('status') == 'pong'
        except Exception as e:
            self.logger.error(f"Ping failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get bot status and open trades.
        
        Returns:
            Bot status information including open trades
        """
        return self._make_request('GET', '/status')
    
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
        params = {
            'pair': pair,
            'timeframe': timeframe,
            'limit': min(limit, 1500)
        }
        
        try:
            response = self._make_request('GET', '/pair_candles', params=params)
            
            if 'data' in response and response['data']:
                df = pd.DataFrame(response['data'])
                
                # Convert timestamp and set as index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Convert to numeric types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
                
                self.logger.info(f"Retrieved {len(df)} candles for {pair} {timeframe}")
                return df
            else:
                self.logger.warning(f"No candle data received for {pair} {timeframe}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to get candles for {pair} {timeframe}: {e}")
            return pd.DataFrame()
    
    def force_enter(self, 
                   pair: str, 
                   side: str = 'long',
                   entry_tag: str = 'gpt_vision_entry') -> Dict[str, Any]:
        """
        Force entry into a position - simplified version without stake amount.
        Let Freqtrade handle position sizing based on its configuration.
        
        Args:
            pair: Trading pair
            side: 'long' or 'short'
            entry_tag: Tag for the entry
            
        Returns:
            Response from force entry command
        """
        data = {
            'pair': pair,
            'side': side,
            'entry_tag': entry_tag
        }
        
        try:
            response = self._make_request('POST', '/forceenter', json=data)
            self.logger.info(f"Force entry executed for {pair} {side}: {response.get('status', 'unknown')}")
            return response
        except Exception as e:
            self.logger.error(f"Failed to force entry for {pair} {side}: {e}")
            raise
    
    def force_exit(self, 
                  pair: str,
                  trade_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Force exit from a position - simplified to exit full position.
        No amount specification - always exits the complete position.
        
        Args:
            pair: Trading pair to exit
            trade_id: Specific trade ID to exit (optional)
            
        Returns:
            Response from force exit command
        """
        data = {}
        
        if trade_id is not None:
            data['tradeid'] = trade_id
        else:
            data['pair'] = pair
        
        try:
            response = self._make_request('POST', '/forceexit', json=data)
            self.logger.info(f"Force exit executed for {pair}: {response.get('status', 'unknown')}")
            return response
        except Exception as e:
            self.logger.error(f"Failed to force exit for {pair}: {e}")
            raise
    
    def get_open_trades(self) -> List[Dict[str, Any]]:
        """
        Get list of currently open trades.
        
        Returns:
            List of open trade dictionaries
        """
        try:
            status = self._make_request('GET', '/status')
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
            return self._make_request('GET', '/profit')
        except Exception as e:
            self.logger.error(f"Failed to get profit info: {e}")
            return {}
    
    def get_performance(self) -> List[Dict[str, Any]]:
        """
        Get performance statistics per pair.
        
        Returns:
            List of performance dictionaries per pair
        """
        try:
            return self._make_request('GET', '/performance')
        except Exception as e:
            self.logger.error(f"Failed to get performance: {e}")
            return []
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """
        Get current price for a trading pair using the most recent candle.
        
        Args:
            pair: Trading pair
            
        Returns:
            Current price or None if unavailable
        """
        try:
            df = self.get_pair_candles(pair, '1m', limit=1)
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception as e:
            self.logger.error(f"Failed to get current price for {pair}: {e}")
        
        return None
    
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
            return True  # Assume dry run if we can't determine
    
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
            
            return {
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
        except Exception as e:
            self.logger.error(f"Failed to get trading summary: {e}")
            return {'error': str(e)}


class TradingOperations:
    """
    High-level trading operations using the Freqtrade API client.
    Provides simplified methods for common trading tasks.
    """
    
    def __init__(self, api_client: FreqtradeAPIClient):
        """
        Initialize trading operations.
        
        Args:
            api_client: Freqtrade API client instance
        """
        self.api = api_client
        self.logger = logging.getLogger(__name__)
    
    def enter_position(self, pair: str, direction: str, reason: str = "gpt_signal") -> bool:
        """
        Enter a position based on trading signal.
        
        Args:
            pair: Trading pair
            direction: 'long' or 'short'
            reason: Reason for entry (used as entry tag)
            
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
            
            # Execute force entry
            response = self.api.force_enter(
                pair=pair,
                side=direction,
                entry_tag=f"{reason}_{direction}"
            )
            
            success = response.get('status') == 'success'
            if success:
                self.logger.info(f"Successfully entered {direction} position for {pair}")
            else:
                self.logger.warning(f"Force entry failed for {pair}: {response}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error entering position for {pair}: {e}")
            return False
    
    def exit_position(self, pair: str, reason: str = "gpt_signal") -> bool:
        """
        Exit position for a trading pair.
        
        Args:
            pair: Trading pair
            reason: Reason for exit
            
        Returns:
            True if successful
        """
        try:
            # Check if position exists
            open_trades = self.api.get_open_trades()
            existing_trade = next((t for t in open_trades if t.get('pair') == pair), None)
            
            if not existing_trade:
                self.logger.info(f"No position exists for {pair}, skipping exit")
                return False
            
            # Execute force exit
            response = self.api.force_exit(
                pair=pair,
                trade_id=existing_trade.get('trade_id')
            )
            
            success = response.get('status') == 'success'
            if success:
                self.logger.info(f"Successfully exited position for {pair}")
            else:
                self.logger.warning(f"Force exit failed for {pair}: {response}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error exiting position for {pair}: {e}")
            return False
    
    def execute_trading_signals(self, signals: Dict[str, bool], pair: str) -> Dict[str, bool]:
        """
        Execute trading signals for a pair.
        
        Args:
            signals: Trading signals dictionary
            pair: Trading pair
            
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
            
            # Process entry signals
            if signals.get('enter_long'):
                results['enter_long_success'] = self.enter_position(pair, 'long', "gpt_bullish")
            elif signals.get('enter_short'):
                results['enter_short_success'] = self.enter_position(pair, 'short', "gpt_bearish")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing trading signals for {pair}: {e}")
            return results
