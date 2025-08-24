"""
API Integration Components
=========================

This module contains API clients and integrations:
- Freqtrade REST API client
- Trading execution controllers
- Live data providers
"""

from .freqtrade_client import FreqtradeAPIClient, FreqtradeConfig, TradingOperations
from .trading_controller import TradingController
