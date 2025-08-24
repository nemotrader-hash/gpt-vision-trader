# 📚 API Reference

## 🎯 **Overview**

This document provides comprehensive API reference for all GPT Vision Trader components. The system is organized into modules for easy integration and extension.

## 📦 **Package Structure**

```python
from gpt_vision_trader import (
    GPTAnalyzer,           # Main GPT analyzer
    ChartGenerator,        # Chart generation  
    DataProcessor,         # Data processing
    TradingConfig,         # Configuration
    FreqtradeAPIClient     # Freqtrade integration
)
```

## 🧠 **Core Module (`gpt_vision_trader.core`)**

### **GPTAnalyzer**

Main class for GPT vision analysis of trading charts.

```python
from gpt_vision_trader.core.gpt_analyzer import GPTAnalyzer

analyzer = GPTAnalyzer(api_key="your_key", model="gpt-4o")
prediction, analysis = await analyzer.analyze_chart("chart.png")
```

#### **Constructor**
```python
GPTAnalyzer(api_key: str, model: str = "gpt-4o")
```

**Parameters:**
- `api_key` (str): OpenAI API key
- `model` (str): GPT model to use (`gpt-4o`, `gpt-4o-mini`)

#### **Methods**

##### `analyze_chart(chart_path: str) -> Tuple[str, Optional[str]]`
Analyze a trading chart and return prediction and analysis.

**Parameters:**
- `chart_path` (str): Path to chart image file

**Returns:**
- `Tuple[str, Optional[str]]`: (prediction, analysis_text)
  - `prediction`: One of `"bullish"`, `"bearish"`, `"neutral"`, or error code
  - `analysis_text`: GPT's analysis of the chart (optional)

**Example:**
```python
prediction, analysis = await analyzer.analyze_chart("btc_chart.png")
print(f"Prediction: {prediction}")
print(f"Analysis: {analysis}")
```

### **LiveGPTAnalyzer**

Enhanced GPT analyzer for real-time trading with signal generation.

```python
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer

live_analyzer = LiveGPTAnalyzer(api_key="your_key", model="gpt-4o")
signals = await live_analyzer.analyze_and_generate_signals("chart.png")
```

#### **Constructor**
```python
LiveGPTAnalyzer(api_key: str, model: str = "gpt-4o")
```

#### **Methods**

##### `analyze_and_generate_signals(chart_path: str) -> Dict[str, Union[str, bool]]`
Analyze chart and generate trading signals.

**Returns:**
```python
{
    'prediction': str,           # GPT prediction
    'analysis': str,            # Analysis text
    'timestamp': str,           # ISO timestamp
    'analysis_count': int,      # Number of analyses performed
    'enter_long': bool,         # Long entry signal
    'exit_long': bool,          # Long exit signal  
    'enter_short': bool,        # Short entry signal
    'exit_short': bool          # Short exit signal
}
```

##### `get_stats() -> Dict`
Get analyzer statistics.

**Returns:**
```python
{
    'analysis_count': int,
    'last_analysis_time': str,
    'cache_size': int,
    'model': str
}
```

### **ChartGenerator**

Generates candlestick charts with technical indicators for GPT analysis.

```python
from gpt_vision_trader.core.chart_generator import ChartGenerator

generator = ChartGenerator(
    output_dir="charts/",
    technical_indicators=indicators
)
chart_path = generator.generate_chart(ohlcv_data, "BTC/USDT Analysis")
```

#### **Constructor**
```python
ChartGenerator(
    output_dir: str,
    technical_indicators: Optional[Dict[str, BaseIndicator]] = None
)
```

**Parameters:**
- `output_dir` (str): Directory to save chart images
- `technical_indicators` (Dict): Dictionary of technical indicators

#### **Methods**

##### `generate_chart(ohlcv_data: pd.DataFrame, title: str, hide_after: Optional[pd.Timestamp] = None, filename: Optional[str] = None) -> str`
Generate a candlestick chart with technical indicators.

**Parameters:**
- `ohlcv_data` (pd.DataFrame): OHLCV data with technical indicators
- `title` (str): Chart title
- `hide_after` (Optional[pd.Timestamp]): Timestamp after which to hide data
- `filename` (Optional[str]): Custom filename (auto-generated if None)

**Returns:**
- `str`: Path to generated chart image

##### `generate_live_chart(visible_data: pd.DataFrame, hidden_placeholder: pd.DataFrame, title: str, pair: str, timeframe: str) -> str`
Generate a live trading chart with hidden future space.

**Parameters:**
- `visible_data` (pd.DataFrame): Historical OHLCV data (visible portion)
- `hidden_placeholder` (pd.DataFrame): Empty future data placeholder
- `title` (str): Chart title
- `pair` (str): Trading pair
- `timeframe` (str): Timeframe

**Returns:**
- `str`: Path to generated chart image

##### `cleanup_old_charts(keep_last_n: int = 10) -> None`
Clean up old chart files to prevent disk space issues.

### **DataProcessor**

Processes OHLCV data for chart generation and analysis.

```python
from gpt_vision_trader.core.data_processor import DataProcessor

processor = DataProcessor(visible_days=6, hidden_days=1)
visible_data, hidden_placeholder = processor.extract_analysis_window(
    ohlcv_data, "15m"
)
```

#### **Constructor**
```python
DataProcessor(visible_days: int = 6, hidden_days: int = 1)
```

#### **Methods**

##### `extract_analysis_window(ohlcv_data: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]`
Extract visible and hidden data windows for analysis.

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: (visible_data, hidden_placeholder)

##### `prepare_metadata(visible_data: pd.DataFrame, chart_path: str, pair: str, timeframe: str, additional_data: Optional[Dict] = None) -> Dict`
Prepare metadata for GPT analysis in backtest-compatible format.

**Returns:**
- `Dict`: Metadata dictionary compatible with GPT analysis system

## 🌐 **API Module (`gpt_vision_trader.api`)**

### **FreqtradeAPIClient**

Simplified Freqtrade REST API client focused on essential trading operations.

```python
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig

config = FreqtradeConfig(base_url="http://localhost:8080")
client = FreqtradeAPIClient(config)
```

#### **Constructor**
```python
FreqtradeAPIClient(config: FreqtradeConfig)
```

#### **Methods**

##### `ping() -> bool`
Test API connection.

**Returns:**
- `bool`: True if API is accessible

##### `get_status() -> Dict[str, Any]`
Get bot status and open trades.

**Returns:**
- `Dict`: Bot status information including open trades

##### `get_pair_candles(pair: str, timeframe: str, limit: int = 500) -> pd.DataFrame`
Get OHLCV candle data for a trading pair.

**Parameters:**
- `pair` (str): Trading pair (e.g., 'BTC/USDT')
- `timeframe` (str): Timeframe (e.g., '15m', '1h', '4h')
- `limit` (int): Number of candles to retrieve (max 1500)

**Returns:**
- `pd.DataFrame`: OHLCV data with datetime index

##### `force_enter(pair: str, side: str = 'long', entry_tag: str = 'gpt_vision_entry') -> Dict[str, Any]`
Force entry into a position (simplified - no stake amount).

**Parameters:**
- `pair` (str): Trading pair
- `side` (str): 'long' or 'short'
- `entry_tag` (str): Tag for the entry

**Returns:**
- `Dict`: Response from force entry command

##### `force_exit(pair: str, trade_id: Optional[int] = None) -> Dict[str, Any]`
Force exit from a position (always exits full position).

**Parameters:**
- `pair` (str): Trading pair to exit
- `trade_id` (Optional[int]): Specific trade ID to exit

**Returns:**
- `Dict`: Response from force exit command

##### `get_open_trades() -> List[Dict[str, Any]]`
Get list of currently open trades.

##### `get_current_price(pair: str) -> Optional[float]`
Get current price for a trading pair.

##### `is_dry_run() -> bool`
Check if bot is running in dry run mode.

##### `get_trading_summary() -> Dict[str, Any]`
Get comprehensive trading summary.

**Returns:**
```python
{
    'bot_state': str,
    'dry_run': bool,
    'open_trades_count': int,
    'max_open_trades': int,
    'stake_currency': str,
    'total_profit': float,
    'profit_ratio': float,
    'trade_count': int,
    'win_rate': float,
    'open_trades': List[Dict]
}
```

### **TradingOperations**

High-level trading operations using the Freqtrade API client.

```python
from gpt_vision_trader.api.freqtrade_client import TradingOperations

trading_ops = TradingOperations(api_client)
success = trading_ops.enter_position("BTC/USDT", "long", "gpt_bullish")
```

#### **Constructor**
```python
TradingOperations(api_client: FreqtradeAPIClient)
```

#### **Methods**

##### `enter_position(pair: str, direction: str, reason: str = "gpt_signal") -> bool`
Enter a position based on trading signal.

##### `exit_position(pair: str, reason: str = "gpt_signal") -> bool`
Exit position for a trading pair.

##### `execute_trading_signals(signals: Dict[str, bool], pair: str) -> Dict[str, bool]`
Execute trading signals for a pair.

**Parameters:**
- `signals` (Dict): Trading signals dictionary
- `pair` (str): Trading pair

**Returns:**
```python
{
    'enter_long_success': bool,
    'exit_long_success': bool,
    'enter_short_success': bool,
    'exit_short_success': bool
}
```

### **TradingController**

Main trading controller that coordinates all trading activities.

```python
from gpt_vision_trader.api.trading_controller import TradingController

controller = TradingController(
    api_client=api_client,
    gpt_analyzer=gpt_analyzer,
    pair="BTC/USDT",
    timeframe="15m"
)
result = await controller.run_trading_cycle()
```

#### **Constructor**
```python
TradingController(
    api_client: FreqtradeAPIClient,
    gpt_analyzer: LiveGPTAnalyzer,
    pair: str,
    timeframe: str,
    visible_days: int = 6,
    hidden_days: int = 1
)
```

#### **Methods**

##### `run_analysis_cycle() -> Optional[Dict]`
Run a complete analysis cycle.

**Returns:**
```python
{
    'cycle_id': int,
    'timestamp': str,
    'pair': str,
    'timeframe': str,
    'chart_path': str,
    'prediction': str,
    'analysis': str,
    'enter_long': bool,
    'exit_long': bool,
    'enter_short': bool,
    'exit_short': bool
}
```

##### `execute_trading_signals(analysis_result: Dict) -> Dict`
Execute trading signals based on analysis results.

##### `run_trading_cycle() -> Optional[Dict]`
Run a complete trading cycle: analyze + execute.

##### `get_status() -> Dict`
Get current controller status.

### **TradingSession**

Manages a complete trading session with multiple cycles.

```python
from gpt_vision_trader.api.trading_controller import TradingSession

session = TradingSession(controller)
await session.run_session(max_cycles=10, cycle_interval_minutes=15)
```

#### **Methods**

##### `run_session(max_cycles: Optional[int] = None, cycle_interval_minutes: int = 15) -> None`
Run a trading session with multiple cycles.

##### `get_session_stats() -> Dict`
Get session statistics.

## 🛠️ **Utils Module (`gpt_vision_trader.utils`)**

### **Technical Indicators**

#### **BaseIndicator** (Abstract Base Class)
```python
from gpt_vision_trader.utils.indicators import BaseIndicator

class CustomIndicator(BaseIndicator):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation
        pass
    
    def plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        # Implementation
        pass
```

#### **SMAIndicator** (Simple Moving Average)
```python
from gpt_vision_trader.utils.indicators import SMAIndicator

sma = SMAIndicator(period=20, color='#ff7f0e', alpha=0.8, linewidth=2.0)
df_with_sma = sma.calculate(ohlcv_df)
```

**Constructor:**
```python
SMAIndicator(
    period: int = 20,
    color: str = '#1f77b4',
    alpha: float = 0.8,
    linewidth: float = 1.0
)
```

#### **RSIIndicator** (Relative Strength Index)
```python
from gpt_vision_trader.utils.indicators import RSIIndicator

rsi = RSIIndicator(
    period=14, 
    overbought=70, 
    oversold=30,
    color='#8757ff'
)
```

#### **MACDIndicator** (Moving Average Convergence Divergence)
```python
from gpt_vision_trader.utils.indicators import MACDIndicator

macd = MACDIndicator(
    fast_period=12,
    slow_period=26,
    signal_period=9
)
```

#### **ATRIndicator** (Average True Range)
```python
from gpt_vision_trader.utils.indicators import ATRIndicator

atr = ATRIndicator(period=14, color='#e67e22')
```

### **Logging Utilities**

#### **setup_logging**
```python
from gpt_vision_trader.utils.logging_utils import setup_logging

setup_logging(
    level=logging.INFO,
    log_file="trading.log",
    log_dir="logs/"
)
```

#### **TradingLogHandler**
```python
from gpt_vision_trader.utils.logging_utils import TradingLogHandler

trading_logger = TradingLogHandler("trading_logs/")

# Log analysis
trading_logger.log_analysis(
    chart_path="chart.png",
    prediction="bullish",
    analysis="Strong uptrend detected",
    pair="BTC/USDT",
    timeframe="15m"
)

# Log trade execution
trading_logger.log_trade_execution(
    action="enter_long",
    pair="BTC/USDT", 
    success=True,
    details={"price": 50000}
)
```

## ⚙️ **Config Module (`gpt_vision_trader.config`)**

### **TradingConfig**

Centralized configuration management with multiple profiles.

```python
from gpt_vision_trader.config.settings import TradingConfig

# Use predefined profiles
config = TradingConfig.from_run_all_settings()  # Default
config = TradingConfig.for_development()        # Dev profile
config = TradingConfig.for_production()         # Prod profile

# Custom configuration
config = TradingConfig(
    symbol="ETH/USDT",
    timeframe="1h",
    gpt_model="gpt-4o-mini",
    analysis_interval_minutes=60
)
```

#### **Class Attributes**
```python
@dataclass
class TradingConfig:
    # Trading parameters
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    visible_days: int = 6
    hidden_days: int = 1
    
    # GPT configuration
    gpt_model: str = "gpt-4o"
    openai_api_key: Optional[str] = None
    
    # Freqtrade API
    freqtrade_url: str = "http://127.0.0.1:8080"
    freqtrade_username: str = "freqtrade"
    freqtrade_password: str = "SuperSecret1!"
    
    # Timing
    analysis_interval_minutes: int = 15
    
    # Directories
    data_dir: str = "gpt_vision_trader/data"
    temp_charts_dir: str = "gpt_vision_trader/data/temp_charts"
    logs_dir: str = "gpt_vision_trader/data/logs"
```

#### **Methods**

##### `get_technical_indicators() -> Dict[str, BaseIndicator]`
Get technical indicators configuration.

##### `validate(require_api_key: bool = True) -> None`
Validate configuration settings.

##### `to_dict() -> Dict`
Convert configuration to dictionary.

##### **Class Methods**

##### `from_run_all_settings() -> TradingConfig`
Create configuration that matches original run_all.py settings.

##### `for_development() -> TradingConfig`
Create configuration optimized for development and testing.

##### `for_production() -> TradingConfig`
Create configuration optimized for production trading.

## 🎮 **Scripts Module (`gpt_vision_trader.scripts`)**

### **Command Line Interface**

#### **Main Entry Point**
```bash
python -m gpt_vision_trader.scripts.run_live_trading [OPTIONS]
```

#### **Available Options**
```bash
# Configuration profiles
--profile {default,dev,prod}

# Trading parameters  
--symbol TEXT              # Trading pair (e.g., BTC/USDT)
--timeframe TEXT           # Timeframe (e.g., 15m, 1h)
--gpt-model TEXT           # GPT model (gpt-4o, gpt-4o-mini)
--analysis-interval INT    # Analysis interval in minutes

# Freqtrade API settings
--freqtrade-url TEXT       # Freqtrade API URL
--freqtrade-username TEXT  # API username
--freqtrade-password TEXT  # API password

# Execution settings
--test                     # Run in test mode (single cycle)
--max-cycles INT          # Maximum number of trading cycles
--verbose, -v             # Enable verbose logging
--log-file TEXT           # Log file path
```

## 🧪 **Testing Module (`gpt_vision_trader.tests`)**

### **Integration Tests**
```python
# Run integration tests
python gpt_vision_trader/tests/test_integration.py

# Or use pytest
pytest gpt_vision_trader/tests/
```

## 📊 **Data Structures**

### **OHLCV DataFrame Format**
```python
# Expected DataFrame structure
df = pd.DataFrame({
    'open': [50000.0, 50100.0, ...],
    'high': [50200.0, 50300.0, ...], 
    'low': [49800.0, 49900.0, ...],
    'close': [50100.0, 50200.0, ...],
    'volume': [1000.0, 1100.0, ...]
}, index=pd.DatetimeIndex([...]))
```

### **Trading Signals Format**
```python
signals = {
    'enter_long': bool,    # Enter long position
    'exit_long': bool,     # Exit long position
    'enter_short': bool,   # Enter short position  
    'exit_short': bool     # Exit short position
}
```

### **Analysis Result Format**
```python
analysis_result = {
    'prediction': str,           # 'bullish', 'bearish', 'neutral'
    'analysis': str,            # GPT analysis text
    'timestamp': str,           # ISO timestamp
    'analysis_count': int,      # Analysis counter
    'enter_long': bool,         # Trading signals...
    'exit_long': bool,
    'enter_short': bool,
    'exit_short': bool
}
```

## 🔧 **Error Handling**

### **Common Exceptions**
```python
# Configuration errors
ValueError: "OpenAI API key is required"
ValueError: "Invalid timeframe: 2m"

# API errors  
requests.RequestException: "Connection refused"
openai.APIError: "Rate limit exceeded"

# Data errors
FileNotFoundError: "Chart file not found"
pd.errors.EmptyDataError: "No OHLCV data available"
```

### **Error Handling Patterns**
```python
try:
    result = await analyzer.analyze_chart(chart_path)
except FileNotFoundError:
    logger.error(f"Chart file not found: {chart_path}")
    return None
except Exception as e:
    logger.error(f"Analysis failed: {e}")
    return None
```

## 🚀 **Usage Examples**

### **Basic Usage**
```python
import asyncio
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer

async def main():
    config = TradingConfig.from_run_all_settings()
    analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
    
    signals = await analyzer.analyze_and_generate_signals("chart.png")
    print(f"Prediction: {signals['prediction']}")

asyncio.run(main())
```

### **Complete Trading System**
```python
import asyncio
from gpt_vision_trader import *

async def run_trading():
    # Configuration
    config = TradingConfig.for_production()
    
    # Initialize components
    freqtrade_config = FreqtradeConfig()
    api_client = FreqtradeAPIClient(freqtrade_config)
    gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
    
    # Create controller
    controller = TradingController(
        api_client=api_client,
        gpt_analyzer=gpt_analyzer,
        pair=config.symbol,
        timeframe=config.timeframe
    )
    
    # Run trading session
    session = TradingSession(controller)
    await session.run_session(max_cycles=10)

asyncio.run(run_trading())
```

This API reference provides comprehensive documentation for integrating and extending the GPT Vision Trader system. All components are designed to be modular and easily customizable for different trading strategies and requirements.
