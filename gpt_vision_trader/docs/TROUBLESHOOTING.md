# 🔧 Troubleshooting Guide

## 🎯 **Quick Diagnostics**

### **System Health Check**
```bash
# 1. Test package installation
python -c "import gpt_vision_trader; print('✅ Package installed')"

# 2. Run integration tests
python gpt_vision_trader/tests/test_integration.py

# 3. Test CLI
python -m gpt_vision_trader.scripts.run_live_trading --help

# 4. Check API key
echo $OPENAI_API_KEY

# 5. Test Freqtrade connection
curl http://127.0.0.1:8080/api/v1/ping
```

## 🚨 **Common Issues & Solutions**

### **1. Installation Issues**

#### **❌ Package Not Found**
```bash
ModuleNotFoundError: No module named 'gpt_vision_trader'
```

**Solutions:**
```bash
# Install in development mode
pip install -e .

# Check if installed
pip list | grep gpt-vision-trader

# Reinstall if needed
pip uninstall gpt-vision-trader
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### **❌ Permission Denied**
```bash
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
```bash
# Use --user flag
pip install --user -e .

# Or create virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .

# On Windows
python -m venv venv
venv\Scripts\activate
pip install -e .
```

#### **❌ Dependency Conflicts**
```bash
ERROR: pip's dependency resolver does not currently consider all the packages
```

**Solutions:**
```bash
# Update pip
pip install --upgrade pip

# Clear cache
pip cache purge

# Install with --force-reinstall
pip install --force-reinstall -e .

# Use conda instead
conda create -n gpt-vision-trader python=3.9
conda activate gpt-vision-trader
pip install -e .
```

### **2. Configuration Issues**

#### **❌ OpenAI API Key Not Found**
```bash
ValueError: OpenAI API key is required
```

**Solutions:**
```bash
# Set environment variable
export OPENAI_API_KEY="your_api_key_here"

# Verify it's set
echo $OPENAI_API_KEY

# Add to shell profile (permanent)
echo 'export OPENAI_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc

# Create .env file
echo 'OPENAI_API_KEY=your_key' > .env
```

#### **❌ Invalid Configuration**
```bash
ValueError: Invalid timeframe: 2m
```

**Solutions:**
```python
# Check valid timeframes
valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# Test configuration
from gpt_vision_trader.config.settings import TradingConfig
config = TradingConfig(timeframe="15m")  # Use valid timeframe
config.validate()
```

#### **❌ Configuration File Issues**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'config.json'
```

**Solutions:**
```python
# Use programmatic configuration instead
from gpt_vision_trader.config.settings import TradingConfig

# Don't rely on config files - use code configuration
config = TradingConfig.from_run_all_settings()
```

### **3. API Connection Issues**

#### **❌ Freqtrade Connection Refused**
```bash
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8080): Max retries exceeded
```

**Solutions:**
```bash
# 1. Check if Freqtrade is running
ps aux | grep freqtrade

# 2. Start Freqtrade with API
freqtrade trade --config config.json --strategy DefaultStrategy

# 3. Test API manually
curl http://127.0.0.1:8080/api/v1/ping

# 4. Check port availability
netstat -an | grep 8080

# 5. Try different port
python -m gpt_vision_trader.scripts.run_live_trading --freqtrade-url http://localhost:8081
```

#### **❌ Authentication Failed**
```bash
requests.exceptions.HTTPError: 401 Client Error: Unauthorized
```

**Solutions:**
```bash
# Check credentials in Freqtrade config.json
{
    "api_server": {
        "username": "freqtrade",
        "password": "SuperSecret1!"
    }
}

# Test with curl
curl -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/status

# Update credentials in command
python -m gpt_vision_trader.scripts.run_live_trading \
    --freqtrade-username your_username \
    --freqtrade-password your_password
```

#### **❌ OpenAI API Errors**
```bash
openai.RateLimitError: Rate limit exceeded
```

**Solutions:**
```bash
# 1. Check API usage at https://platform.openai.com/usage

# 2. Use cheaper model
python -m gpt_vision_trader.scripts.run_live_trading --gpt-model gpt-4o-mini

# 3. Reduce frequency
python -m gpt_vision_trader.scripts.run_live_trading --analysis-interval 30

# 4. Check API key validity
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('API key valid!')
"
```

### **4. Data Issues**

#### **❌ No OHLCV Data**
```bash
ValueError: OHLCV data is empty
```

**Solutions:**
```bash
# 1. Check if trading pair exists on exchange
curl "http://127.0.0.1:8080/api/v1/pair_candles?pair=BTC/USDT&timeframe=15m&limit=10"

# 2. Try different pair
python -m gpt_vision_trader.scripts.run_live_trading --symbol ETH/USDT

# 3. Check exchange configuration in Freqtrade
# Make sure exchange is properly configured and connected

# 4. Check whitelist in Freqtrade config
{
    "exchange": {
        "pair_whitelist": ["BTC/USDT", "ETH/USDT"]
    }
}
```

#### **❌ Chart Generation Failed**
```bash
FileNotFoundError: Chart file not found
```

**Solutions:**
```bash
# 1. Check directory permissions
ls -la gpt_vision_trader/data/temp_charts/

# 2. Create directories manually
mkdir -p gpt_vision_trader/data/temp_charts

# 3. Check disk space
df -h

# 4. Test chart generation
python -c "
from gpt_vision_trader.tests.test_integration import test_chart_generator
test_chart_generator()
"
```

### **5. Trading Execution Issues**

#### **❌ Force Entry Failed**
```bash
Force entry failed: {'status': 'error', 'reason': 'Insufficient balance'}
```

**Solutions:**
```bash
# 1. Check available balance
curl -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/balance

# 2. Reduce stake amount in Freqtrade config
{
    "stake_amount": 10  # Reduce from 100 to 10
}

# 3. Check dry run mode
{
    "dry_run": true,
    "dry_run_wallet": 1000
}

# 4. Check exchange connectivity
curl -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/status
```

#### **❌ Position Already Exists**
```bash
Force entry failed: {'status': 'error', 'reason': 'Position already exists'}
```

**Solutions:**
```bash
# 1. Check open trades
curl -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/status

# 2. Close existing position
curl -X POST -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/forceexit \
    -H "Content-Type: application/json" \
    -d '{"pair": "BTC/USDT"}'

# 3. Increase max_open_trades
{
    "max_open_trades": 3  # Allow multiple positions
}
```

### **6. Performance Issues**

#### **❌ Slow Analysis**
```bash
Analysis taking too long...
```

**Solutions:**
```bash
# 1. Use faster model
python -m gpt_vision_trader.scripts.run_live_trading --gpt-model gpt-4o-mini

# 2. Reduce visible days
config = TradingConfig(visible_days=3)  # Instead of 6

# 3. Increase analysis interval
python -m gpt_vision_trader.scripts.run_live_trading --analysis-interval 30

# 4. Check internet connection
ping api.openai.com
```

#### **❌ High API Costs**
```bash
OpenAI charges are too high
```

**Solutions:**
```bash
# 1. Use gpt-4o-mini (90% cheaper)
python -m gpt_vision_trader.scripts.run_live_trading --gpt-model gpt-4o-mini

# 2. Reduce analysis frequency
python -m gpt_vision_trader.scripts.run_live_trading --analysis-interval 60

# 3. Limit daily cycles
python -m gpt_vision_trader.scripts.run_live_trading --max-cycles 24

# 4. Use longer timeframes
python -m gpt_vision_trader.scripts.run_live_trading --timeframe 1h
```

### **7. Import Errors**

#### **❌ Module Import Failed**
```python
ImportError: cannot import name 'TradingConfig' from 'gpt_vision_trader.config.settings'
```

**Solutions:**
```bash
# 1. Reinstall package
pip uninstall gpt-vision-trader
pip install -e .

# 2. Check file exists
ls -la gpt_vision_trader/config/settings.py

# 3. Check Python path
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())
from gpt_vision_trader.config.settings import TradingConfig
print('Import successful!')
"

# 4. Use absolute imports
from gpt_vision_trader.config.settings import TradingConfig
```

#### **❌ Circular Import**
```python
ImportError: cannot import name 'X' from partially initialized module
```

**Solutions:**
```python
# Use lazy imports
def get_analyzer():
    from gpt_vision_trader.core.gpt_analyzer import GPTAnalyzer
    return GPTAnalyzer

# Or import at function level
def analyze_chart():
    from gpt_vision_trader.core.gpt_analyzer import GPTAnalyzer
    analyzer = GPTAnalyzer(api_key, model)
    # ...
```

## 🔍 **Debugging Tools**

### **Enable Debug Logging**
```bash
# Verbose logging
python -m gpt_vision_trader.scripts.run_live_trading --verbose

# Log to file
python -m gpt_vision_trader.scripts.run_live_trading --log-file debug.log

# Check log files
tail -f gpt_vision_trader/data/logs/gpt_vision_trader_*.log
tail -f gpt_vision_trader/data/trading_logs/trading_*.jsonl
```

### **Test Individual Components**
```python
# Test configuration
from gpt_vision_trader.config.settings import TradingConfig
config = TradingConfig.from_run_all_settings()
config.validate()
print("✅ Configuration OK")

# Test API client
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
client = FreqtradeAPIClient(FreqtradeConfig())
print(f"API connection: {client.ping()}")

# Test GPT analyzer (requires API key)
import asyncio
from gpt_vision_trader.core.gpt_analyzer import GPTAnalyzer
async def test_gpt():
    analyzer = GPTAnalyzer("your_key", "gpt-4o-mini")
    # Create test chart first...
asyncio.run(test_gpt())
```

### **Network Debugging**
```bash
# Check network connectivity
ping api.openai.com
ping 127.0.0.1

# Check port availability
netstat -an | grep 8080
lsof -i :8080

# Test HTTP endpoints
curl -v http://127.0.0.1:8080/api/v1/ping
curl -v -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/status
```

### **File System Debugging**
```bash
# Check file permissions
ls -la gpt_vision_trader/
ls -la gpt_vision_trader/data/

# Check disk space
df -h

# Check directory structure
tree gpt_vision_trader/ -L 3

# Create missing directories
mkdir -p gpt_vision_trader/data/{temp_charts,logs,trading_logs}
```

## 🧪 **Testing & Validation**

### **Run Integration Tests**
```bash
# Full integration test
python gpt_vision_trader/tests/test_integration.py

# Expected output
✅ Configuration: PASSED
✅ Technical Indicators: PASSED  
✅ Data Processor: PASSED
✅ Chart Generator: PASSED
✅ GPT Analyzer: PASSED
✅ Freqtrade API Client: PASSED
```

### **Test Mode Validation**
```bash
# Test without real trading
python -m gpt_vision_trader.scripts.run_live_trading --test

# Should complete one analysis cycle safely
```

### **Configuration Validation**
```python
# Test all profiles
from gpt_vision_trader.config.settings import TradingConfig

configs = [
    TradingConfig.from_run_all_settings(),
    TradingConfig.for_development(),
    TradingConfig.for_production()
]

for config in configs:
    try:
        config.validate(require_api_key=False)
        print(f"✅ {config.__class__.__name__} valid")
    except Exception as e:
        print(f"❌ {config.__class__.__name__} invalid: {e}")
```

## 📊 **Performance Monitoring**

### **Monitor System Resources**
```bash
# CPU and memory usage
top -p $(pgrep -f "gpt_vision_trader")

# Disk usage
du -h gpt_vision_trader/data/

# Network usage
netstat -i
```

### **Monitor API Usage**
```bash
# OpenAI API usage
# Check at: https://platform.openai.com/usage

# Freqtrade API logs
tail -f /path/to/freqtrade/logs/freqtrade.log

# Trading activity
tail -f gpt_vision_trader/data/trading_logs/trading_$(date +%Y%m%d).jsonl
```

### **Monitor Trading Performance**
```bash
# Check Freqtrade UI
open http://localhost:8080

# Check trade history
curl -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/trades

# Check profit/loss
curl -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/profit
```

## 🆘 **Getting Help**

### **Before Asking for Help**
1. ✅ Run integration tests
2. ✅ Check logs for error messages
3. ✅ Try test mode first
4. ✅ Verify API keys and connections
5. ✅ Check this troubleshooting guide

### **Information to Include**
```bash
# System information
python --version
pip list | grep -E "(gpt-vision-trader|openai|pandas|matplotlib)"

# Error logs
tail -20 gpt_vision_trader/data/logs/gpt_vision_trader_*.log

# Configuration (without API keys)
python -c "
from gpt_vision_trader.config.settings import TradingConfig
config = TradingConfig.from_run_all_settings()
print(config.to_dict())
"
```

### **Support Channels**
- **GitHub Issues**: [Repository Issues](https://github.com/yourusername/gpt-vision-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gpt-vision-trader/discussions)
- **Documentation**: Check all docs in `gpt_vision_trader/docs/`

## 🔧 **Advanced Troubleshooting**

### **Environment Isolation**
```bash
# Create clean environment
conda create -n gpt-vision-clean python=3.9
conda activate gpt-vision-clean
pip install -e .

# Or with venv
python -m venv clean_env
source clean_env/bin/activate  # Linux/Mac
clean_env\Scripts\activate     # Windows
pip install -e .
```

### **Dependency Analysis**
```bash
# Check dependency tree
pip install pipdeptree
pipdeptree --packages gpt-vision-trader

# Check for conflicts
pip check

# Update all dependencies
pip install --upgrade -r requirements.txt
```

### **Code Debugging**
```python
# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Or use ipdb
import ipdb; ipdb.set_trace()
```

## ✅ **Recovery Checklist**

When things go wrong, follow this recovery process:

1. **Stop all processes**
   ```bash
   # Stop trading
   pkill -f "gpt_vision_trader"
   
   # Stop Freqtrade
   pkill -f "freqtrade"
   ```

2. **Check system health**
   ```bash
   python gpt_vision_trader/tests/test_integration.py
   ```

3. **Restart services**
   ```bash
   # Start Freqtrade
   freqtrade trade --config config.json --strategy DefaultStrategy
   
   # Test connection
   curl http://127.0.0.1:8080/api/v1/ping
   
   # Start trading in test mode
   python -m gpt_vision_trader.scripts.run_live_trading --test
   ```

4. **Verify everything works**
   ```bash
   # Check logs
   tail -f gpt_vision_trader/data/logs/gpt_vision_trader_*.log
   
   # Monitor trading
   tail -f gpt_vision_trader/data/trading_logs/trading_*.jsonl
   ```

**Remember: When in doubt, start with test mode and work your way up to live trading!** 🛡️
