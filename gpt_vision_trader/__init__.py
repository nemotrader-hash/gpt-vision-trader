"""
GPT Vision Trader - AI-Powered Trading System
============================================

A sophisticated trading system that uses GPT vision analysis of candlestick charts
to make trading decisions. Integrates with Freqtrade for live trading execution.

Main Components:
- Core: Chart generation, GPT analysis, and data processing
- API: Freqtrade REST API integration and trade execution
- Utils: Configuration, logging, and helper functions
- Strategies: Freqtrade strategy implementations
"""

__version__ = "1.0.0"
__author__ = "GPT Vision Trader Team"

# Core imports for easy access
from .core.gpt_analyzer import GPTAnalyzer
from .core.chart_generator import ChartGenerator
from .core.data_processor import DataProcessor
from .config.settings import TradingConfig
from .api.freqtrade_client import FreqtradeAPIClient
