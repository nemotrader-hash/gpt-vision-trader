# 🚀 GPT Vision Trader

> **AI-Powered Trading System using GPT Vision Analysis of Candlestick Charts**

A sophisticated trading system that leverages OpenAI's GPT Vision API to analyze candlestick charts and make intelligent trading decisions through Freqtrade integration.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 **What It Does**

GPT Vision Trader analyzes candlestick charts using AI and executes trades automatically:

1. **📊 Fetches real-time OHLCV data** from Freqtrade
2. **📈 Generates candlestick charts** with technical indicators (SMA, RSI, MACD, ATR)
3. **🤖 Analyzes charts with GPT Vision** to predict market movements
4. **⚡ Executes trades automatically** via Freqtrade's REST API
5. **📝 Logs all decisions** for analysis and monitoring

## 🏗️ **Architecture**

```
gpt_vision_trader/
├── core/                    # Core trading logic
│   ├── gpt_analyzer.py     # GPT vision analysis
│   ├── chart_generator.py   # Chart generation with indicators
│   └── data_processor.py    # Data processing utilities
├── api/                     # External integrations
│   ├── freqtrade_client.py  # Freqtrade REST API client
│   └── trading_controller.py # Main trading controller
├── utils/                   # Utilities
│   ├── indicators.py        # Technical indicators (SMA, RSI, etc.)
│   └── logging_utils.py     # Logging system
├── config/                  # Configuration
│   └── settings.py          # Centralized settings & profiles
├── scripts/                 # Entry points
│   └── run_live_trading.py  # Main CLI application
└── tests/                   # Test suite
    └── test_integration.py  # Integration tests
```

## ⚡ **Quick Start**

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/gpt-vision-trader.git
cd gpt-vision-trader

# Install package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### **2. Configuration**

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"
```

### **3. Start Freqtrade with REST API**

Add to your Freqtrade `config.json`:
```json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "freqtrade",
        "password": "SuperSecret1!",
        "jwt_secret_key": "somethingrandom"
    }
}
```

Start Freqtrade:
```bash
freqtrade trade --config config.json --strategy DefaultStrategy
```

### **4. Test the System**

```bash
# Run integration tests
python gpt_vision_trader/tests/test_integration.py

# Test live trading (safe mode)
python -m gpt_vision_trader.scripts.run_live_trading --test
```

### **5. Start Live Trading**

```bash
# Default settings (BTC/USDT, 15m, gpt-4o)
python -m gpt_vision_trader.scripts.run_live_trading

# Development profile (cheaper, faster)
python -m gpt_vision_trader.scripts.run_live_trading --profile dev

# Production profile (optimized)
python -m gpt_vision_trader.scripts.run_live_trading --profile prod
```

## 🎛️ **Configuration Profiles**

### **Development Profile** (`--profile dev`)
- **GPT Model**: gpt-4o-mini (90% cheaper)
- **Analysis Interval**: 30 minutes
- **Visible Days**: 3 (faster testing)
- **Cost**: ~$0.10-0.30/day

### **Production Profile** (`--profile prod`)
- **GPT Model**: gpt-4o (balanced performance/cost)
- **Analysis Interval**: 15 minutes
- **Visible Days**: 6 (matches backtesting exactly)
- **Cost**: ~$1-3/day

### **Default Profile** (matches backtesting)
- **Symbol**: BTC/USDT
- **Timeframe**: 15m
- **GPT Model**: gpt-4o
- **Technical Indicators**: SMA20, SMA50

## 🎮 **Command Line Usage**

### **Basic Commands**
```bash
# Test mode (safe, no real trading)
python -m gpt_vision_trader.scripts.run_live_trading --test

# Live trading with default settings
python -m gpt_vision_trader.scripts.run_live_trading

# Help and options
python -m gpt_vision_trader.scripts.run_live_trading --help
```

### **Advanced Usage**
```bash
# Custom symbol and timeframe
python -m gpt_vision_trader.scripts.run_live_trading \
    --symbol ETH/USDT \
    --timeframe 1h \
    --gpt-model gpt-4o-mini

# Limited cycles for testing
python -m gpt_vision_trader.scripts.run_live_trading \
    --max-cycles 5 \
    --analysis-interval 30

# Custom Freqtrade API settings
python -m gpt_vision_trader.scripts.run_live_trading \
    --freqtrade-url http://localhost:8080 \
    --freqtrade-username myuser \
    --freqtrade-password mypass

# Verbose logging to file
python -m gpt_vision_trader.scripts.run_live_trading \
    --verbose \
    --log-file trading.log
```

## 💻 **Programmatic Usage**

```python
import asyncio
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader.api.trading_controller import TradingController, TradingSession

async def main():
    # Create configuration
    config = TradingConfig.for_production()
    
    # Initialize components
    freqtrade_config = FreqtradeConfig()
    api_client = FreqtradeAPIClient(freqtrade_config)
    gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
    
    # Create trading controller
    controller = TradingController(
        api_client=api_client,
        gpt_analyzer=gpt_analyzer,
        pair=config.symbol,
        timeframe=config.timeframe
    )
    
    # Run trading session
    session = TradingSession(controller)
    await session.run_session(max_cycles=10, cycle_interval_minutes=15)

# Run
asyncio.run(main())
```

## 📊 **Trading Logic**

### **Analysis Process**
1. **Fetch OHLCV data** from Freqtrade API
2. **Extract analysis window** (6 days visible + 1 day hidden)
3. **Generate candlestick chart** with technical indicators
4. **Analyze with GPT Vision** to predict market direction
5. **Generate trading signals** based on prediction

### **Signal Generation**
- **Bullish Prediction** → `forceenter` long position
- **Bearish Prediction** → `forceenter` short position  
- **Neutral Prediction** → `forceexit` all positions

### **Position Management**
- **One position per pair** maximum
- **Full position exits** only (no partial closes)
- **Freqtrade handles position sizing** automatically
- **Entry tags** include prediction reason

## 🛡️ **Risk Management**

### **Built-in Safety Features**
- **Dry run support** - Test without real money
- **Position limits** - Maximum one position per pair
- **Error handling** - Safe fallbacks on API failures
- **Graceful shutdown** - Ctrl+C handling
- **Comprehensive logging** - Full audit trail

### **Recommended Settings**
```json
{
    "dry_run": true,           // Start with paper trading
    "max_open_trades": 1,      // Limit concurrent positions
    "stake_amount": 100,       // Fixed stake per trade
    "stoploss": -0.10          // 10% stop loss
}
```

## 📈 **Performance Expectations**

The system maintains **100% compatibility** with backtesting:
- **Same GPT analysis pipeline**
- **Identical chart generation**  
- **Same technical indicators**
- **Same prediction logic**

**Expected result**: Live performance should closely match backtest performance (accounting for real-time market conditions).

## 💰 **Cost Management**

### **GPT API Costs** (per day at 15min intervals)
- **gpt-4o**: ~$1-3/day (96 analyses × $0.01-0.03)
- **gpt-4o-mini**: ~$0.10-0.30/day (96 analyses × $0.001-0.003)

### **Cost Optimization**
```bash
# Use cheaper model
python -m gpt_vision_trader.scripts.run_live_trading --gpt-model gpt-4o-mini

# Reduce frequency  
python -m gpt_vision_trader.scripts.run_live_trading --analysis-interval 30

# Limit daily cycles
python -m gpt_vision_trader.scripts.run_live_trading --max-cycles 48
```

## 📝 **Monitoring & Logging**

### **Log Files**
- **System logs**: `gpt_vision_trader/data/logs/gpt_vision_trader_YYYYMMDD.log`
- **Trading logs**: `gpt_vision_trader/data/trading_logs/trading_YYYYMMDD.jsonl`
- **Analysis logs**: `gpt_vision_trader/data/analysis_logs/analysis_YYYYMMDD.jsonl`

### **Real-time Monitoring**
```bash
# Watch trading activity
tail -f gpt_vision_trader/data/trading_logs/trading_$(date +%Y%m%d).jsonl

# Monitor system logs
tail -f gpt_vision_trader/data/logs/gpt_vision_trader_$(date +%Y%m%d).log

# Freqtrade web interface
open http://localhost:8080
```

## 🧪 **Testing**

### **Integration Tests**
```bash
# Run all integration tests
python gpt_vision_trader/tests/test_integration.py

# Expected output:
# ✅ Configuration: PASSED
# ✅ Technical Indicators: PASSED  
# ✅ Data Processor: PASSED
# ✅ Chart Generator: PASSED
# ✅ GPT Analyzer: PASSED
# ✅ Freqtrade API Client: PASSED
```

### **Live Testing**
```bash
# Test mode (no real trading)
python -m gpt_vision_trader.scripts.run_live_trading --test

# Development profile testing
python -m gpt_vision_trader.scripts.run_live_trading --profile dev --max-cycles 3
```

## 🔧 **Troubleshooting**

### **Common Issues**

**Connection refused to Freqtrade API**
```bash
# Check if Freqtrade is running
ps aux | grep freqtrade

# Verify API configuration
curl http://127.0.0.1:8080/api/v1/ping
```

**OpenAI API key not found**
```bash
# Set environment variable
export OPENAI_API_KEY="your_key_here"

# Verify it's set
echo $OPENAI_API_KEY
```

**Import errors**
```bash
# Install in development mode
pip install -e .

# Or install dependencies
pip install -r requirements.txt
```

## 📚 **Documentation**

- **[RESTRUCTURE_SUMMARY.md](RESTRUCTURE_SUMMARY.md)** - Complete system overview
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migration from old structure
- **[Freqtrade API Docs](https://www.freqtrade.io/en/stable/rest-api/)** - API reference
- **[OpenAI Vision API](https://platform.openai.com/docs/guides/vision)** - GPT Vision documentation

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `python gpt_vision_trader/tests/test_integration.py`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ **Disclaimer**

**This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Never trade with money you cannot afford to lose. Past performance does not guarantee future results. Always test thoroughly in dry-run mode before live trading.**

## 🆘 **Support**

- **Issues**: [GitHub Issues](https://github.com/nemotrader-hash/gpt-vision-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nemotrader-hash/gpt-vision-trader/discussions)
- **Documentation**: Check the docs folder for detailed guides

---

**Made with ❤️ by the GPT Vision Trader Team**

*Happy Trading! 🚀📈*
