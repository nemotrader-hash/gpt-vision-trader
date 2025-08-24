#!/usr/bin/env python3
"""
Integration Test Suite
=====================

Comprehensive integration tests for the restructured GPT Vision Trading system.
Tests all components working together without actual trading.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import GPTAnalyzer, LiveGPTAnalyzer
from gpt_vision_trader.core.chart_generator import ChartGenerator
from gpt_vision_trader.core.data_processor import DataProcessor
from gpt_vision_trader.utils.indicators import SMAIndicator, RSIIndicator
from gpt_vision_trader.utils.logging_utils import setup_logging


def setup_test_logging():
    """Setup logging for testing."""
    setup_logging(level=logging.INFO, log_file="gpt_vision_trader/data/logs/test_integration.log")


def test_configuration():
    """Test configuration management."""
    print("\n" + "="*50)
    print("Testing Configuration Management")
    print("="*50)
    
    try:
        # Test default configuration
        config = TradingConfig.from_run_all_settings()
        print(f"✅ Default configuration created")
        print(f"   Symbol: {config.symbol}")
        print(f"   Timeframe: {config.timeframe}")
        print(f"   GPT Model: {config.gpt_model}")
        
        # Test configuration validation (without requiring API key for tests)
        config.validate(require_api_key=False)
        print("✅ Configuration validation passed")
        
        # Test technical indicators
        indicators = config.get_technical_indicators()
        print(f"✅ Technical indicators loaded: {len(indicators)} indicators")
        
        # Test different profiles
        dev_config = TradingConfig.for_development()
        prod_config = TradingConfig.for_production()
        print(f"✅ Configuration profiles created (dev: {dev_config.gpt_model}, prod: {prod_config.gpt_model})")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_technical_indicators():
    """Test technical indicators."""
    print("\n" + "="*50)
    print("Testing Technical Indicators")
    print("="*50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=100, freq='15T')
        np.random.seed(42)
        
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        high_prices = close_prices + np.random.rand(100) * 200
        low_prices = close_prices - np.random.rand(100) * 200
        open_prices = close_prices + np.random.randn(100) * 50
        volumes = np.random.rand(100) * 1000
        
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
        
        # Test SMA indicator
        sma_indicator = SMAIndicator(period=20)
        df_with_sma = sma_indicator.calculate(df)
        
        if 'SMA_20' in df_with_sma.columns:
            print("✅ SMA indicator calculation successful")
        else:
            print("❌ SMA indicator failed")
            return False
        
        # Test RSI indicator
        rsi_indicator = RSIIndicator(period=14)
        df_with_rsi = rsi_indicator.calculate(df_with_sma)
        
        if 'RSI_14' in df_with_rsi.columns:
            print("✅ RSI indicator calculation successful")
        else:
            print("❌ RSI indicator failed")
            return False
        
        print(f"✅ Technical indicators test passed")
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False


def test_data_processor():
    """Test data processing functionality."""
    print("\n" + "="*50)
    print("Testing Data Processor")
    print("="*50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=1000, freq='15T')
        np.random.seed(42)
        
        close_prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
        
        df = pd.DataFrame({
            'open': close_prices + np.random.randn(1000) * 50,
            'high': close_prices + np.random.rand(1000) * 200,
            'low': close_prices - np.random.rand(1000) * 200,
            'close': close_prices,
            'volume': np.random.rand(1000) * 1000
        }, index=dates)
        
        # Test data processor
        processor = DataProcessor(visible_days=6, hidden_days=1)
        
        visible_data, hidden_placeholder = processor.extract_analysis_window(df, '15m')
        
        print(f"✅ Analysis window extracted: {len(visible_data)} visible, {len(hidden_placeholder)} hidden")
        
        # Test metadata preparation
        metadata = processor.prepare_metadata(
            visible_data, 
            "test_chart.png",
            "BTC/USDT",
            "15m"
        )
        
        if "test_chart.png" in metadata:
            print("✅ Metadata preparation successful")
        else:
            print("❌ Metadata preparation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False


def test_chart_generator():
    """Test chart generation."""
    print("\n" + "="*50)
    print("Testing Chart Generator")
    print("="*50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=200, freq='15T')
        np.random.seed(42)
        
        close_prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        
        df = pd.DataFrame({
            'open': close_prices + np.random.randn(200) * 50,
            'high': close_prices + np.random.rand(200) * 200,
            'low': close_prices - np.random.rand(200) * 200,
            'close': close_prices,
            'volume': np.random.rand(200) * 1000
        }, index=dates)
        
        # Create chart generator with indicators
        config = TradingConfig.from_run_all_settings()
        indicators = config.get_technical_indicators()
        
        chart_generator = ChartGenerator(
            output_dir="gpt_vision_trader/data/temp_charts",
            technical_indicators=indicators
        )
        
        # Generate chart
        chart_path = chart_generator.generate_chart(
            ohlcv_data=df,
            title="Test Chart - Integration Test",
            filename="test_integration_chart.png"
        )
        
        if os.path.exists(chart_path):
            print(f"✅ Chart generated successfully: {chart_path}")
            return True
        else:
            print("❌ Chart generation failed")
            return False
        
    except Exception as e:
        print(f"❌ Chart generator test failed: {e}")
        return False


async def test_gpt_analyzer():
    """Test GPT analysis (if API key available)."""
    print("\n" + "="*50)
    print("Testing GPT Analyzer")
    print("="*50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OpenAI API key not available - skipping GPT tests")
        return True
    
    try:
        # Test basic GPT analyzer
        analyzer = GPTAnalyzer(api_key, "gpt-4o-mini")  # Use cheaper model for testing
        
        # Create a test chart (should exist from previous test)
        test_chart_path = "gpt_vision_trader/data/temp_charts/test_integration_chart.png"
        
        if not os.path.exists(test_chart_path):
            print("⚠️ Test chart not found - skipping GPT analysis test")
            return True
        
        # Analyze chart
        prediction, analysis = await analyzer.analyze_chart(test_chart_path)
        
        print(f"✅ GPT analysis completed")
        print(f"   Prediction: {prediction}")
        print(f"   Analysis: {analysis[:100] if analysis else 'No analysis'}...")
        
        # Test live analyzer
        live_analyzer = LiveGPTAnalyzer(api_key, "gpt-4o-mini")
        signals = await live_analyzer.analyze_and_generate_signals(test_chart_path)
        
        print(f"✅ Live GPT analysis completed")
        print(f"   Signals: {signals}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPT analyzer test failed: {e}")
        return False


def test_freqtrade_api_client():
    """Test Freqtrade API client (without actual connection)."""
    print("\n" + "="*50)
    print("Testing Freqtrade API Client")
    print("="*50)
    
    try:
        # Test configuration
        freqtrade_config = FreqtradeConfig()
        client = FreqtradeAPIClient(freqtrade_config)
        
        print(f"✅ Freqtrade API client initialized")
        print(f"   URL: {freqtrade_config.base_url}")
        print(f"   Username: {freqtrade_config.username}")
        
        # Test connection (expected to fail without running Freqtrade)
        try:
            is_connected = client.ping()
            if is_connected:
                print("✅ Freqtrade API connection successful")
                
                # Test additional methods if connected
                status = client.get_status()
                print(f"   Bot state: {status.get('state', 'unknown')}")
                
                # Test getting candle data
                df = client.get_pair_candles('BTC/USDT', '15m', limit=10)
                if not df.empty:
                    print(f"   Candle data retrieved: {len(df)} candles")
            else:
                print("⚠️ Freqtrade API connection failed (expected if Freqtrade not running)")
                
        except Exception as e:
            print(f"⚠️ Freqtrade API connection failed: {e} (expected if not running)")
        
        return True
        
    except Exception as e:
        print(f"❌ Freqtrade API client test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests."""
    print("GPT Vision Trader - Restructured Integration Tests")
    print("=" * 60)
    
    # Setup logging
    setup_test_logging()
    
    tests = [
        ("Configuration", test_configuration),
        ("Technical Indicators", test_technical_indicators),
        ("Data Processor", test_data_processor),
        ("Chart Generator", test_chart_generator),
        ("GPT Analyzer", test_gpt_analyzer),
        ("Freqtrade API Client", test_freqtrade_api_client),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status_icon = "✅" if passed else "❌"
        status_text = "PASSED" if passed else "FAILED"
        print(f"{status_icon} {test_name}: {status_text}")
    
    overall_success = all(results.values())
    
    if overall_success:
        print("\n🎉 All integration tests passed!")
        print("\nThe restructured system is ready for use!")
        print("\nNext steps:")
        print("1. Run: python -m gpt_vision_trader.scripts.run_live_trading --test")
        print("2. For live trading: python -m gpt_vision_trader.scripts.run_live_trading")
    else:
        print("\n⚠️ Some integration tests failed")
        print("Please check the failed components")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)
