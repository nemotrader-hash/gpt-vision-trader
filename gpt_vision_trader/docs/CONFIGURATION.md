# ⚙️ Configuration Guide

## 🎯 **Overview**

GPT Vision Trader uses a centralized configuration system with multiple profiles for different use cases. All configuration is managed through the `TradingConfig` class in `gpt_vision_trader/config/settings.py`.

## 📋 **Configuration Profiles**

### **Default Profile** - Matches Backtesting
```python
from gpt_vision_trader.config.settings import TradingConfig

config = TradingConfig.from_run_all_settings()
```

**Settings:**
- **Symbol**: BTC/USDT
- **Timeframe**: 15m
- **Visible Days**: 6 (matches backtesting exactly)
- **Hidden Days**: 1 (prediction horizon)
- **GPT Model**: gpt-4o
- **Analysis Interval**: 15 minutes
- **Technical Indicators**: SMA20, SMA50

### **Development Profile** - Fast & Cheap
```python
config = TradingConfig.for_development()
```

**Settings:**
- **GPT Model**: gpt-4o-mini (90% cheaper)
- **Analysis Interval**: 30 minutes (less frequent)
- **Visible Days**: 3 (faster testing)
- **Cost**: ~$0.10-0.30/day

### **Production Profile** - Optimized
```python
config = TradingConfig.for_production()
```

**Settings:**
- **GPT Model**: gpt-4o (balanced performance/cost)
- **Analysis Interval**: 15 minutes (standard frequency)
- **Visible Days**: 6 (matches backtesting exactly)
- **Cost**: ~$1-3/day

## 🔧 **Configuration Parameters**

### **Trading Parameters**
```python
@dataclass
class TradingConfig:
    # Trading pair and timeframe
    symbol: str = "BTC/USDT"              # Trading pair
    timeframe: str = "15m"                # Candlestick timeframe
    
    # Analysis window settings - CRITICAL: Must match backtest
    visible_days: int = 6                 # Days of visible data
    hidden_days: int = 1                  # Prediction horizon
    
    # GPT model configuration
    gpt_model: str = "gpt-4o"            # OpenAI model
    openai_api_key: Optional[str] = None  # API key (from env)
    
    # Trading cycle settings
    analysis_interval_minutes: int = 15   # Analysis frequency
```

### **Freqtrade API Settings**
```python
    # Freqtrade API configuration
    freqtrade_url: str = "http://127.0.0.1:8080"
    freqtrade_username: str = "freqtrade"
    freqtrade_password: str = "SuperSecret1!"
    freqtrade_timeout: int = 30
```

### **Data Directories**
```python
    # File system paths
    data_dir: str = "gpt_vision_trader/data"
    temp_charts_dir: str = "gpt_vision_trader/data/temp_charts"
    logs_dir: str = "gpt_vision_trader/data/logs"
```

## 🎨 **Technical Indicators Configuration**

### **Default Indicators** (Matches Backtesting)
```python
def get_technical_indicators(self) -> Dict[str, BaseIndicator]:
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
    }
```

### **Extended Indicators** (Optional)
```python
# Uncomment in settings.py to enable additional indicators
"RSI14": RSIIndicator(
    period=14,
    color='#8757ff', 
    alpha=0.8,
    linewidth=1.0,
    overbought=70,
    oversold=30
),
"MACD": MACDIndicator(
    fast_period=12,
    slow_period=26,
    signal_period=9,
    color='#2ecc71', 
    signal_color='#e74c3c',  
    macd_color='#3498db',
    alpha=0.8,
    linewidth=1.0
),
"ATR14": ATRIndicator(
    period=14,
    color='#e67e22', 
    alpha=0.8,
    linewidth=1.0
)
```

## 🌍 **Environment Variables**

### **Required Variables**
```bash
# OpenAI API Key (required)
export OPENAI_API_KEY="your_openai_api_key_here"
```

### **Optional Variables**
```bash
# Freqtrade API settings (optional, uses defaults if not set)
export FREQTRADE_URL="http://127.0.0.1:8080"
export FREQTRADE_USERNAME="freqtrade"
export FREQTRADE_PASSWORD="SuperSecret1!"
```

### **Using .env File**
Create `.env` file in project root:
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
FREQTRADE_URL=http://127.0.0.1:8080
FREQTRADE_USERNAME=freqtrade
FREQTRADE_PASSWORD=SuperSecret1!
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

## 🎛️ **Command Line Configuration**

### **Profile Selection**
```bash
# Use development profile
python -m gpt_vision_trader.scripts.run_live_trading --profile dev

# Use production profile  
python -m gpt_vision_trader.scripts.run_live_trading --profile prod

# Use default profile (no flag needed)
python -m gpt_vision_trader.scripts.run_live_trading
```

### **Parameter Overrides**
```bash
# Override specific parameters
python -m gpt_vision_trader.scripts.run_live_trading \
    --symbol ETH/USDT \
    --timeframe 1h \
    --gpt-model gpt-4o-mini \
    --analysis-interval 30 \
    --freqtrade-url http://localhost:8081 \
    --freqtrade-username myuser \
    --freqtrade-password mypass
```

## 🔧 **Custom Configuration**

### **Creating Custom Config**
```python
# custom_config.py
from gpt_vision_trader.config.settings import TradingConfig

# Custom configuration for ETH trading
eth_config = TradingConfig(
    symbol="ETH/USDT",
    timeframe="1h", 
    visible_days=7,
    hidden_days=2,
    gpt_model="gpt-4o-mini",
    analysis_interval_minutes=60
)

# Validate configuration
eth_config.validate()

# Use in your application
from gpt_vision_trader.api.trading_controller import TradingController
# ... initialize with eth_config
```

### **Multiple Trading Pairs**
```python
# multi_pair_config.py
configs = {
    "BTC/USDT": TradingConfig(
        symbol="BTC/USDT",
        timeframe="15m",
        gpt_model="gpt-4o"
    ),
    "ETH/USDT": TradingConfig(
        symbol="ETH/USDT", 
        timeframe="15m",
        gpt_model="gpt-4o-mini"  # Use cheaper model for ETH
    ),
    "ADA/USDT": TradingConfig(
        symbol="ADA/USDT",
        timeframe="30m",         # Less frequent for smaller cap
        analysis_interval_minutes=30
    )
}
```

## 🎯 **Freqtrade Configuration**

### **Basic Freqtrade Config**
```json
{
    "trading_mode": "spot",
    "dry_run": true,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "max_open_trades": 1,
    "timeframe": "15m",
    
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "freqtrade",
        "password": "SuperSecret1!",
        "jwt_secret_key": "somethingrandom"
    },
    
    "exchange": {
        "name": "binance",
        "key": "your_exchange_key",
        "secret": "your_exchange_secret",
        "ccxt_config": {},
        "ccxt_async_config": {},
        "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
        "pair_blacklist": []
    }
}
```

### **Security Configuration**
```json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",    // Localhost only
        "listen_port": 8080,
        "verbosity": "error",                 // Minimal logging
        "enable_openapi": false,              // Disable Swagger UI
        "jwt_secret_key": "use_secrets.token_hex()",
        "CORS_origins": [],                   // No CORS
        "username": "secure_username",        // Change default
        "password": "very_secure_password123!", // Strong password
        "ws_token": "secure_websocket_token"
    }
}
```

## 💰 **Cost Optimization Configuration**

### **Budget-Friendly Setup**
```python
budget_config = TradingConfig(
    symbol="BTC/USDT",
    timeframe="1h",                    # Longer timeframe
    gpt_model="gpt-4o-mini",          # Cheapest model
    analysis_interval_minutes=60,      # Less frequent analysis
    visible_days=3                     # Shorter analysis window
)
# Cost: ~$0.05-0.15/day
```

### **High-Frequency Setup**
```python
hf_config = TradingConfig(
    symbol="BTC/USDT", 
    timeframe="5m",                    # Short timeframe
    gpt_model="gpt-4o",               # Better model
    analysis_interval_minutes=5,       # Very frequent
    visible_days=2                     # Shorter for speed
)
# Cost: ~$10-30/day
```

## 🧪 **Testing Configuration**

### **Test Mode Settings**
```python
test_config = TradingConfig(
    symbol="BTC/USDT",
    timeframe="15m",
    gpt_model="gpt-4o-mini",          # Cheap for testing
    analysis_interval_minutes=60,      # Less frequent
    visible_days=1,                    # Minimal data
    hidden_days=1
)
```

### **Backtesting Compatibility**
```python
# CRITICAL: Use these exact settings to match backtesting
backtest_config = TradingConfig(
    symbol="BTC/USDT",      # Same as backtest
    timeframe="15m",        # Same as backtest  
    visible_days=6,         # MUST match backtest exactly
    hidden_days=1,          # MUST match backtest exactly
    gpt_model="gpt-4o",     # Same model as backtest
    analysis_interval_minutes=15
)
```

## ✅ **Configuration Validation**

### **Automatic Validation**
```python
config = TradingConfig.from_run_all_settings()

# Validate configuration (raises ValueError if invalid)
config.validate()

# Optional: Skip API key validation for testing
config.validate(require_api_key=False)
```

### **Validation Rules**
- **OpenAI API key** must be provided (unless `require_api_key=False`)
- **Visible days** must be positive
- **Hidden days** must be positive  
- **Analysis interval** must be positive
- **Timeframe** must be valid (`1m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`)
- **Symbol** must be in format `BASE/QUOTE` (e.g., `BTC/USDT`)

### **Configuration Export**
```python
config = TradingConfig.for_production()

# Export to dictionary
config_dict = config.to_dict()
print(json.dumps(config_dict, indent=2))

# Save to file
with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
```

## 🔍 **Configuration Debugging**

### **Check Current Configuration**
```python
from gpt_vision_trader.config.settings import TradingConfig

config = TradingConfig.from_run_all_settings()

print("Current Configuration:")
print(f"Symbol: {config.symbol}")
print(f"Timeframe: {config.timeframe}")
print(f"GPT Model: {config.gpt_model}")
print(f"Analysis Interval: {config.analysis_interval_minutes} minutes")
print(f"OpenAI API Key: {'Set' if config.openai_api_key else 'Not Set'}")
```

### **Validate Settings**
```bash
# Test configuration via CLI
python -c "
from gpt_vision_trader.config.settings import TradingConfig
config = TradingConfig.from_run_all_settings()
try:
    config.validate()
    print('✅ Configuration is valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

## 📊 **Performance Configuration**

### **Optimal Settings by Use Case**

#### **Day Trading** (High Frequency)
```python
day_trading_config = TradingConfig(
    timeframe="5m",
    analysis_interval_minutes=5,
    visible_days=1,
    gpt_model="gpt-4o"
)
```

#### **Swing Trading** (Medium Frequency)  
```python
swing_trading_config = TradingConfig(
    timeframe="1h", 
    analysis_interval_minutes=60,
    visible_days=7,
    gpt_model="gpt-4o"
)
```

#### **Position Trading** (Low Frequency)
```python
position_trading_config = TradingConfig(
    timeframe="4h",
    analysis_interval_minutes=240, 
    visible_days=14,
    gpt_model="gpt-4o-mini"
)
```

## 🚨 **Important Notes**

### **Backtesting Compatibility**
To ensure live trading matches backtesting results:
- **Use exact same `visible_days` and `hidden_days`**
- **Use same technical indicators**
- **Use same GPT model**
- **Use same symbol and timeframe**

### **Cost Management**
- **gpt-4o-mini**: 90% cheaper than gpt-4o
- **Longer intervals**: Reduce API call frequency
- **Shorter visible days**: Smaller images, lower costs

### **Security**
- **Never commit API keys** to version control
- **Use environment variables** or `.env` files
- **Change default Freqtrade credentials**
- **Limit API access** to localhost only

**Your configuration is the foundation of successful trading. Choose settings that match your risk tolerance, budget, and trading style!** 🎯
