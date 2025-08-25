#!/usr/bin/env python3
"""
Configuration Settings
======================

Centralized configuration management for the GPT Vision Trading system.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

from ..utils.indicators import BaseIndicator, SMAIndicator, RSIIndicator, MACDIndicator, ATRIndicator


@dataclass
class TradingConfig:
    """
    Main configuration class for GPT Vision Trading.
    Maintains compatibility with the original run_all.py settings.
    """
    
    # Trading pair and timeframe
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    
    # Analysis window settings - CRITICAL: Must match backtest settings
    visible_days: int = 6
    hidden_days: int = 1  # Prediction horizon
    indicator_buffer_days: int = 2  # Additional days for indicator pre-calculation
    
    # GPT model configuration
    gpt_model: str = "gpt-4o"  # Default to newer, more cost-effective model
    openai_api_key: Optional[str] = None
    
    # Freqtrade API settings
    freqtrade_url: str = "http://127.0.0.1:8080"
    freqtrade_username: str = "freqtrade"
    freqtrade_password: str = "SuperSecret1!"
    freqtrade_timeout: int = 30
    
    # Trading cycle settings
    analysis_interval_minutes: int = 15  # How often to run analysis
    
    # Data directories
    data_dir: str = "gpt_vision_trader/data"
    temp_charts_dir: str = "gpt_vision_trader/data/temp_charts"
    logs_dir: str = "gpt_vision_trader/data/logs"
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        import logging
        
        # Get OpenAI API key from environment if not provided
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Debug: Log API key status (without exposing the key)
        if self.openai_api_key:
            logging.info(f"✅ OpenAI API key loaded: {self.openai_api_key[:8]}...{self.openai_api_key[-4:]}")
        else:
            logging.warning("❌ OpenAI API key not found in environment or config")
            logging.warning("   Please set: export OPENAI_API_KEY='your_key_here'")
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_charts_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def get_technical_indicators(self) -> Dict[str, BaseIndicator]:
        """
        Get technical indicators configuration.
        MUST match the original run_all.py settings exactly.
        """
        return {
            "SMA20": SMAIndicator(
                period=20,
                color='#ff7f0e',  
                alpha=0.8,
                linewidth=2.0
            ),
            "SMA50": SMAIndicator(
                period=50,
                color='#8757ff',  
                alpha=0.8,
                linewidth=2.0
            ),
            # Uncomment these if they were used in backtesting
            # "RSI14": RSIIndicator(
            #     period=14,
            #     color='#8757ff', 
            #     alpha=0.8,
            #     linewidth=1.0,
            #     overbought=70,
            #     oversold=30
            # ),
            # "MACD": MACDIndicator(
            #     fast_period=12,
            #     slow_period=26,
            #     signal_period=9,
            #     color='#2ecc71', 
            #     signal_color='#e74c3c',  
            #     macd_color='#3498db',
            #     alpha=0.8,
            #     linewidth=1.0
            # ),
            # "ATR14": ATRIndicator(
            #     period=14,
            #     color='#e67e22', 
            #     alpha=0.8,
            #     linewidth=1.0
            # )
        }
    
    @classmethod
    def from_run_all_settings(cls) -> 'TradingConfig':
        """
        Create configuration that matches the original run_all.py settings.
        Ensures consistency between backtesting and live trading.
        """
        return cls(
            symbol="BTC/USDT",
            timeframe="15m", 
            visible_days=6,
            hidden_days=1,
            indicator_buffer_days=2,
            gpt_model="gpt-4o",  # More cost-effective than gpt-4.1
            analysis_interval_minutes=15
        )
    
    @classmethod
    def for_development(cls) -> 'TradingConfig':
        """
        Create configuration optimized for development and testing.
        """
        return cls(
            symbol="BTC/USDT",
            timeframe="15m",
            visible_days=3,  # Shorter for faster testing
            hidden_days=1,
            indicator_buffer_days=1,  # Smaller buffer for dev
            gpt_model="gpt-4o-mini",  # Cheapest option
            analysis_interval_minutes=30  # Less frequent for development
        )
    
    @classmethod
    def for_production(cls) -> 'TradingConfig':
        """
        Create configuration optimized for production trading.
        """
        return cls(
            symbol="BTC/USDT",
            timeframe="15m",
            visible_days=6,  # Match backtesting exactly
            hidden_days=1,
            indicator_buffer_days=2,
            gpt_model="gpt-4o",  # Good balance of performance and cost
            analysis_interval_minutes=15  # Standard frequency
        )
    
    def validate(self, require_api_key: bool = True) -> None:
        """
        Validate configuration settings.
        
        Args:
            require_api_key: Whether to require OpenAI API key (default True)
        
        Raises:
            ValueError: If configuration is invalid
        """
        if require_api_key and not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.visible_days <= 0:
            raise ValueError("Visible days must be positive")
        
        if self.hidden_days <= 0:
            raise ValueError("Hidden days must be positive")
        
        if self.indicator_buffer_days < 0:
            raise ValueError("Indicator buffer days must be non-negative")
        
        if self.analysis_interval_minutes <= 0:
            raise ValueError("Analysis interval must be positive")
        
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {self.timeframe}")
        
        if not self.symbol or '/' not in self.symbol:
            raise ValueError("Invalid symbol format (should be like BTC/USDT)")
    
    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'visible_days': self.visible_days,
            'hidden_days': self.hidden_days,
            'indicator_buffer_days': self.indicator_buffer_days,
            'gpt_model': self.gpt_model,
            'analysis_interval_minutes': self.analysis_interval_minutes,
            'freqtrade_url': self.freqtrade_url,
            'freqtrade_username': self.freqtrade_username,
            'data_dir': self.data_dir
        }


# Pre-configured instances for easy access
DEFAULT_CONFIG = TradingConfig.from_run_all_settings()
DEV_CONFIG = TradingConfig.for_development()
PROD_CONFIG = TradingConfig.for_production()
