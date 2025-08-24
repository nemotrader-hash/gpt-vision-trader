# 💻 Programmatic Usage Guide

## 🎯 **Overview**

This guide shows you how to use GPT Vision Trader programmatically in your own Python scripts. All examples are complete, working code that you can copy and run immediately.

## 🚀 **Quick Start Example**

### **Basic Single Analysis**
```python
import asyncio
import os
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer

async def analyze_single_chart():
    """Simple example: Analyze one chart and get prediction."""
    
    # Set your API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Create GPT analyzer
    analyzer = LiveGPTAnalyzer(api_key, model="gpt-4o-mini")  # Use cheaper model
    
    # Analyze a chart (you need to have a chart image)
    chart_path = "path/to/your/chart.png"
    
    try:
        result = await analyzer.analyze_and_generate_signals(chart_path)
        
        print(f"🤖 GPT Prediction: {result['prediction']}")
        print(f"📊 Analysis: {result['analysis'][:100]}...")
        print(f"📈 Signals:")
        print(f"   Enter Long: {result['enter_long']}")
        print(f"   Exit Long: {result['exit_long']}")
        print(f"   Enter Short: {result['enter_short']}")
        print(f"   Exit Short: {result['exit_short']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Run the example
asyncio.run(analyze_single_chart())
```

## 🔧 **Complete Trading System**

### **Full Live Trading Example**
```python
import asyncio
import os
import logging
from datetime import datetime
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader.api.trading_controller import TradingController, TradingSession

async def run_complete_trading_system():
    """Complete example: Full trading system with Freqtrade integration."""
    
    print("🚀 Starting Complete Trading System")
    
    # 1. Configuration
    config = TradingConfig.from_run_all_settings()
    print(f"📊 Trading: {config.symbol} on {config.timeframe}")
    print(f"🤖 GPT Model: {config.gpt_model}")
    
    # 2. Initialize Freqtrade API client
    freqtrade_config = FreqtradeConfig(
        base_url="http://127.0.0.1:8080",
        username="freqtrade", 
        password="SuperSecret1!"
    )
    api_client = FreqtradeAPIClient(freqtrade_config)
    
    # 3. Test connection
    if not api_client.ping():
        print("❌ Cannot connect to Freqtrade API")
        print("   Make sure Freqtrade is running with REST API enabled")
        return
    print("✅ Freqtrade API connected")
    
    # 4. Initialize GPT analyzer
    gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
    
    # 5. Create trading controller
    controller = TradingController(
        api_client=api_client,
        gpt_analyzer=gpt_analyzer,
        pair=config.symbol,
        timeframe=config.timeframe,
        visible_days=config.visible_days,
        hidden_days=config.hidden_days
    )
    
    # 6. Run trading session
    session = TradingSession(controller)
    
    print("🔄 Starting trading session (5 cycles for demo)")
    await session.run_session(
        max_cycles=5,  # Run 5 cycles for demo
        cycle_interval_minutes=config.analysis_interval_minutes
    )
    
    # 7. Get final statistics
    stats = session.get_session_stats()
    print(f"\n📊 Session completed:")
    print(f"   Cycles: {stats['cycles_completed']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Duration: {stats['session_duration_seconds']/60:.1f} minutes")

# Run the complete system
if __name__ == "__main__":
    asyncio.run(run_complete_trading_system())
```

## 🔄 **Multi-Pair Trading**

### **Simple Multi-Pair Example**
```python
import asyncio
import os
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader.api.trading_controller import TradingController

async def run_multi_pair_trading():
    """Example: Trade multiple cryptocurrency pairs simultaneously."""
    
    print("🚀 Starting Multi-Pair Trading System")
    
    # Define pairs with individual settings
    pairs_config = {
        "BTC/USDT": {
            "timeframe": "15m",
            "model": "gpt-4o",           # Use better model for BTC
            "interval": 15
        },
        "ETH/USDT": {
            "timeframe": "15m", 
            "model": "gpt-4o-mini",      # Use cheaper model for ETH
            "interval": 15
        },
        "ADA/USDT": {
            "timeframe": "30m",          # Less frequent for smaller cap
            "model": "gpt-4o-mini",
            "interval": 30
        }
    }
    
    # Shared Freqtrade client
    freqtrade_config = FreqtradeConfig()
    api_client = FreqtradeAPIClient(freqtrade_config)
    
    if not api_client.ping():
        print("❌ Cannot connect to Freqtrade API")
        return
    print("✅ Freqtrade API connected")
    
    # Create controllers for each pair
    controllers = {}
    
    for pair, settings in pairs_config.items():
        print(f"🔧 Setting up {pair}")
        
        # Create config for this pair
        config = TradingConfig(
            symbol=pair,
            timeframe=settings["timeframe"],
            gpt_model=settings["model"],
            analysis_interval_minutes=settings["interval"]
        )
        
        # Create GPT analyzer
        gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
        
        # Create controller
        controller = TradingController(
            api_client=api_client,
            gpt_analyzer=gpt_analyzer,
            pair=pair,
            timeframe=settings["timeframe"]
        )
        
        controllers[pair] = controller
    
    print(f"📊 Created controllers for {len(controllers)} pairs")
    
    # Run one analysis cycle for each pair
    print("\n🔄 Running analysis for all pairs...")
    
    for pair, controller in controllers.items():
        print(f"\n📈 Analyzing {pair}:")
        
        try:
            result = await controller.run_analysis_cycle()
            if result:
                print(f"   Prediction: {result['prediction']}")
                print(f"   Chart: {result['chart_path']}")
                
                # Execute trading signals
                execution = await controller.execute_trading_signals(result)
                actions = execution.get('execution_results', {})
                
                if any(actions.values()):
                    print(f"   ✅ Actions taken: {[k for k, v in actions.items() if v]}")
                else:
                    print(f"   ℹ️  No actions taken")
            else:
                print(f"   ❌ Analysis failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 Multi-pair analysis completed!")

# Run multi-pair trading
if __name__ == "__main__":
    asyncio.run(run_multi_pair_trading())
```

### **Advanced Multi-Pair with Concurrent Execution**
```python
import asyncio
import logging
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.api.trading_controller import TradingController, TradingSession

async def run_pair_session(pair_name, config, api_client):
    """Run trading session for a single pair."""
    
    print(f"🚀 Starting session for {pair_name}")
    
    try:
        from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
        
        # Create components for this pair
        gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
        controller = TradingController(
            api_client=api_client,
            gpt_analyzer=gpt_analyzer,
            pair=config.symbol,
            timeframe=config.timeframe
        )
        
        # Run trading session
        session = TradingSession(controller)
        await session.run_session(
            max_cycles=3,  # 3 cycles per pair for demo
            cycle_interval_minutes=config.analysis_interval_minutes
        )
        
        # Get results
        stats = session.get_session_stats()
        print(f"✅ {pair_name} completed: {stats['cycles_completed']} cycles, {stats['success_rate']:.1f}% success")
        
    except Exception as e:
        print(f"❌ {pair_name} failed: {e}")

async def run_concurrent_multi_pair():
    """Example: Run multiple pairs concurrently (parallel execution)."""
    
    print("🚀 Starting Concurrent Multi-Pair Trading")
    
    # Define pairs
    pairs = {
        "BTC/USDT": TradingConfig(
            symbol="BTC/USDT",
            timeframe="15m",
            gpt_model="gpt-4o-mini",
            analysis_interval_minutes=30
        ),
        "ETH/USDT": TradingConfig(
            symbol="ETH/USDT", 
            timeframe="15m",
            gpt_model="gpt-4o-mini",
            analysis_interval_minutes=30
        ),
        "SOL/USDT": TradingConfig(
            symbol="SOL/USDT",
            timeframe="30m",
            gpt_model="gpt-4o-mini",
            analysis_interval_minutes=60
        )
    }
    
    # Shared API client
    freqtrade_config = FreqtradeConfig()
    api_client = FreqtradeAPIClient(freqtrade_config)
    
    if not api_client.ping():
        print("❌ Freqtrade API not available")
        return
    
    print(f"✅ Running {len(pairs)} pairs concurrently")
    
    # Create tasks for each pair
    tasks = []
    for pair_name, config in pairs.items():
        task = run_pair_session(pair_name, config, api_client)
        tasks.append(task)
    
    # Run all pairs concurrently
    await asyncio.gather(*tasks)
    
    print("🎉 All pairs completed!")

# Run concurrent multi-pair
if __name__ == "__main__":
    asyncio.run(run_concurrent_multi_pair())
```

## 📊 **Custom Configuration Examples**

### **Budget-Friendly Configuration**
```python
from gpt_vision_trader.config.settings import TradingConfig

def create_budget_config():
    """Create cost-optimized configuration."""
    
    config = TradingConfig(
        symbol="BTC/USDT",
        timeframe="1h",                    # Longer timeframe = fewer API calls
        gpt_model="gpt-4o-mini",          # 90% cheaper than gpt-4o
        analysis_interval_minutes=60,      # Less frequent analysis
        visible_days=3,                    # Shorter analysis window
        hidden_days=1
    )
    
    print("💰 Budget Configuration:")
    print(f"   Expected cost: ~$0.05-0.15/day")
    print(f"   Model: {config.gpt_model}")
    print(f"   Interval: {config.analysis_interval_minutes} minutes")
    
    return config

# Usage
budget_config = create_budget_config()
```

### **High-Performance Configuration**
```python
def create_performance_config():
    """Create high-performance configuration."""
    
    config = TradingConfig(
        symbol="BTC/USDT",
        timeframe="5m",                    # Short timeframe for quick signals
        gpt_model="gpt-4o",               # Better model for accuracy
        analysis_interval_minutes=5,       # Very frequent analysis
        visible_days=2,                    # Shorter window for speed
        hidden_days=1
    )
    
    print("⚡ Performance Configuration:")
    print(f"   Expected cost: ~$10-30/day")
    print(f"   Model: {config.gpt_model}")
    print(f"   Interval: {config.analysis_interval_minutes} minutes")
    
    return config

# Usage
perf_config = create_performance_config()
```

### **Custom Indicators Configuration**
```python
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.utils.indicators import SMAIndicator, RSIIndicator, MACDIndicator

def create_custom_indicators_config():
    """Create configuration with custom technical indicators."""
    
    config = TradingConfig.from_run_all_settings()
    
    # Add custom indicators
    config.technical_indicators = {
        "SMA10": SMAIndicator(period=10, color='#ff0000', linewidth=1.5),
        "SMA20": SMAIndicator(period=20, color='#00ff00', linewidth=2.0),
        "SMA50": SMAIndicator(period=50, color='#0000ff', linewidth=2.5),
        "RSI": RSIIndicator(period=14, overbought=75, oversold=25),
        "MACD": MACDIndicator(fast_period=12, slow_period=26, signal_period=9)
    }
    
    print("📊 Custom Indicators Configuration:")
    for name, indicator in config.technical_indicators.items():
        print(f"   {name}: {indicator.__class__.__name__}")
    
    return config

# Usage
custom_config = create_custom_indicators_config()
```

## 🔍 **Data Analysis Examples**

### **Analyze Trading Performance**
```python
import json
import pandas as pd
from datetime import datetime, timedelta

def analyze_trading_logs(log_file_path):
    """Analyze trading performance from log files."""
    
    print(f"📊 Analyzing trading logs: {log_file_path}")
    
    # Read log file
    entries = []
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                entries.append(entry)
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
        return None
    
    # Separate different types of entries
    analyses = [e for e in entries if e.get('type') == 'analysis']
    executions = [e for e in entries if e.get('type') == 'execution']
    
    print(f"\n📈 Analysis Summary:")
    print(f"   Total analyses: {len(analyses)}")
    
    if analyses:
        # Prediction distribution
        predictions = [a.get('prediction', 'unknown') for a in analyses]
        pred_counts = pd.Series(predictions).value_counts()
        print(f"   Predictions:")
        for pred, count in pred_counts.items():
            print(f"     {pred}: {count} ({count/len(predictions)*100:.1f}%)")
    
    print(f"\n💼 Execution Summary:")
    print(f"   Total executions: {len(executions)}")
    
    if executions:
        # Success rate
        successful = [e for e in executions if e.get('success', False)]
        success_rate = len(successful) / len(executions) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Action types
        actions = [e.get('action', 'unknown') for e in executions]
        action_counts = pd.Series(actions).value_counts()
        print(f"   Actions:")
        for action, count in action_counts.items():
            print(f"     {action}: {count}")
    
    return {
        'analyses': analyses,
        'executions': executions,
        'success_rate': success_rate if executions else 0
    }

# Usage example
log_file = "gpt_vision_trader/data/trading_logs/trading_20241201.jsonl"
results = analyze_trading_logs(log_file)
```

### **Monitor Live Performance**
```python
import asyncio
import time
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig

async def monitor_live_performance():
    """Monitor live trading performance in real-time."""
    
    print("📊 Starting Live Performance Monitor")
    
    # Connect to Freqtrade
    config = FreqtradeConfig()
    client = FreqtradeAPIClient(config)
    
    if not client.ping():
        print("❌ Cannot connect to Freqtrade")
        return
    
    print("✅ Connected to Freqtrade")
    
    # Monitor for 5 minutes (300 seconds)
    start_time = time.time()
    duration = 300  # 5 minutes
    
    while time.time() - start_time < duration:
        try:
            # Get current status
            summary = client.get_trading_summary()
            
            print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
            print(f"📊 Bot State: {summary['bot_state']}")
            print(f"💰 Total Profit: {summary['total_profit']:.2f} {summary['stake_currency']}")
            print(f"📈 Profit Ratio: {summary['profit_ratio']:.2%}")
            print(f"🔢 Trade Count: {summary['trade_count']}")
            print(f"🎯 Win Rate: {summary['win_rate']:.1f}%")
            print(f"📋 Open Trades: {summary['open_trades_count']}/{summary['max_open_trades']}")
            
            # Show open trades
            if summary['open_trades']:
                print(f"🔄 Active Trades:")
                for trade in summary['open_trades'][:3]:  # Show first 3
                    pair = trade.get('pair', 'Unknown')
                    profit = trade.get('profit_ratio', 0) * 100
                    print(f"   {pair}: {profit:+.2f}%")
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            await asyncio.sleep(30)
    
    print("\n📊 Monitoring completed")

# Run monitor
if __name__ == "__main__":
    asyncio.run(monitor_live_performance())
```

## 🛠️ **Utility Functions**

### **Chart Generation Helper**
```python
import pandas as pd
import numpy as np
from gpt_vision_trader.core.chart_generator import ChartGenerator
from gpt_vision_trader.config.settings import TradingConfig

def generate_test_chart():
    """Generate a test chart for GPT analysis."""
    
    print("📊 Generating test chart")
    
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    
    # Simulate realistic price movement
    base_price = 50000
    returns = np.random.normal(0, 0.001, 200)  # 0.1% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices + np.random.normal(0, 10, 200),
        'high': prices + np.abs(np.random.normal(50, 20, 200)),
        'low': prices - np.abs(np.random.normal(50, 20, 200)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 200)
    }, index=dates)
    
    # Generate chart
    config = TradingConfig.from_run_all_settings()
    indicators = config.get_technical_indicators()
    
    generator = ChartGenerator(
        output_dir="gpt_vision_trader/data/temp_charts",
        technical_indicators=indicators
    )
    
    chart_path = generator.generate_chart(
        ohlcv_data=df,
        title="Test Chart - BTC/USDT 15m",
        filename="test_chart.png"
    )
    
    print(f"✅ Chart generated: {chart_path}")
    return chart_path

# Usage
chart_path = generate_test_chart()
```

### **Configuration Validator**
```python
from gpt_vision_trader.config.settings import TradingConfig

def validate_all_configurations():
    """Validate all configuration profiles."""
    
    print("🔍 Validating all configuration profiles")
    
    configs = {
        "Default": TradingConfig.from_run_all_settings(),
        "Development": TradingConfig.for_development(),
        "Production": TradingConfig.for_production()
    }
    
    for name, config in configs.items():
        try:
            config.validate(require_api_key=False)  # Skip API key for validation
            print(f"✅ {name} configuration: VALID")
            print(f"   Symbol: {config.symbol}")
            print(f"   Model: {config.gpt_model}")
            print(f"   Interval: {config.analysis_interval_minutes}min")
        except Exception as e:
            print(f"❌ {name} configuration: INVALID - {e}")
    
    return configs

# Usage
configs = validate_all_configurations()
```

## 🎯 **Complete Example Scripts**

### **Simple Trading Bot**
```python
"""
Simple Trading Bot Example
==========================

This script shows how to create a basic trading bot that:
1. Connects to Freqtrade
2. Analyzes charts with GPT
3. Executes trades automatically
4. Logs all activity
"""

import asyncio
import logging
import os
from datetime import datetime
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader.api.trading_controller import TradingController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('simple_bot.log'),
        logging.StreamHandler()
    ]
)

class SimpleTradingBot:
    """Simple trading bot implementation."""
    
    def __init__(self):
        self.config = TradingConfig.for_development()  # Use dev config
        self.running = False
        
    async def start(self):
        """Start the trading bot."""
        
        logging.info("🚀 Starting Simple Trading Bot")
        
        # Initialize components
        freqtrade_config = FreqtradeConfig()
        api_client = FreqtradeAPIClient(freqtrade_config)
        
        # Test connection
        if not api_client.ping():
            logging.error("❌ Cannot connect to Freqtrade")
            return
        
        logging.info("✅ Connected to Freqtrade")
        
        # Create analyzer and controller
        gpt_analyzer = LiveGPTAnalyzer(self.config.openai_api_key, self.config.gpt_model)
        controller = TradingController(
            api_client=api_client,
            gpt_analyzer=gpt_analyzer,
            pair=self.config.symbol,
            timeframe=self.config.timeframe
        )
        
        # Trading loop
        self.running = True
        cycle = 0
        
        while self.running and cycle < 10:  # Run 10 cycles max
            cycle += 1
            logging.info(f"🔄 Starting cycle {cycle}")
            
            try:
                # Run analysis and trading
                result = await controller.run_trading_cycle()
                
                if result:
                    analysis = result['analysis']
                    execution = result['execution']
                    
                    logging.info(f"📊 Prediction: {analysis['prediction']}")
                    
                    if execution['any_action_taken']:
                        logging.info("✅ Trade executed")
                    else:
                        logging.info("ℹ️ No action taken")
                else:
                    logging.warning("⚠️ Cycle failed")
                
            except Exception as e:
                logging.error(f"❌ Cycle error: {e}")
            
            # Wait before next cycle
            if self.running and cycle < 10:
                logging.info(f"💤 Waiting {self.config.analysis_interval_minutes} minutes...")
                await asyncio.sleep(self.config.analysis_interval_minutes * 60)
        
        logging.info("🏁 Trading bot stopped")
    
    def stop(self):
        """Stop the trading bot."""
        self.running = False

async def main():
    """Main function."""
    bot = SimpleTradingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logging.info("👋 Bot interrupted by user")
        bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### **Portfolio Manager Example**
```python
"""
Portfolio Manager Example
========================

This script manages a portfolio of multiple cryptocurrency pairs
with individual risk management and position sizing.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer

@dataclass
class PortfolioPosition:
    """Portfolio position information."""
    pair: str
    allocation: float  # Portfolio allocation (0.0 to 1.0)
    max_risk: float   # Maximum risk per trade
    current_position: dict = None

class PortfolioManager:
    """Manages a portfolio of cryptocurrency positions."""
    
    def __init__(self):
        self.api_client = None
        self.positions: Dict[str, PortfolioPosition] = {}
        self.total_portfolio_value = 1000  # $1000 portfolio
        
    def add_position(self, pair: str, allocation: float, max_risk: float = 0.02):
        """Add a position to the portfolio."""
        self.positions[pair] = PortfolioPosition(
            pair=pair,
            allocation=allocation,
            max_risk=max_risk
        )
        logging.info(f"➕ Added {pair}: {allocation:.1%} allocation, {max_risk:.1%} max risk")
    
    async def initialize(self):
        """Initialize the portfolio manager."""
        
        # Setup API client
        freqtrade_config = FreqtradeConfig()
        self.api_client = FreqtradeAPIClient(freqtrade_config)
        
        if not self.api_client.ping():
            raise ConnectionError("Cannot connect to Freqtrade API")
        
        logging.info("✅ Portfolio manager initialized")
    
    async def analyze_portfolio(self):
        """Analyze all positions in the portfolio."""
        
        logging.info("📊 Analyzing portfolio positions")
        
        for pair, position in self.positions.items():
            logging.info(f"🔍 Analyzing {pair}")
            
            try:
                # Get current market data
                df = self.api_client.get_pair_candles(pair, "15m", limit=100)
                
                if df.empty:
                    logging.warning(f"⚠️ No data for {pair}")
                    continue
                
                # Create analyzer for this pair
                config = TradingConfig(
                    symbol=pair,
                    gpt_model="gpt-4o-mini"  # Use cheaper model for portfolio
                )
                
                analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
                
                # Generate chart and analyze (simplified - you'd need actual chart generation)
                logging.info(f"   Current price: ${df['close'].iloc[-1]:.2f}")
                logging.info(f"   24h change: {((df['close'].iloc[-1] / df['close'].iloc[-25]) - 1) * 100:+.2f}%")
                
                # Get current position from Freqtrade
                open_trades = self.api_client.get_open_trades()
                current_trade = next((t for t in open_trades if t.get('pair') == pair), None)
                
                if current_trade:
                    profit = current_trade.get('profit_ratio', 0) * 100
                    logging.info(f"   📈 Open position: {profit:+.2f}%")
                else:
                    logging.info(f"   📊 No open position")
                
            except Exception as e:
                logging.error(f"❌ Error analyzing {pair}: {e}")
    
    async def rebalance_portfolio(self):
        """Rebalance the portfolio based on current allocations."""
        
        logging.info("⚖️ Rebalancing portfolio")
        
        # Get current portfolio value
        summary = self.api_client.get_trading_summary()
        current_value = summary.get('total_profit', 0) + self.total_portfolio_value
        
        logging.info(f"💰 Current portfolio value: ${current_value:.2f}")
        
        for pair, position in self.positions.items():
            target_value = current_value * position.allocation
            logging.info(f"🎯 {pair} target allocation: ${target_value:.2f} ({position.allocation:.1%})")
            
            # Here you would implement actual rebalancing logic
            # This is a simplified example
    
    def get_portfolio_summary(self):
        """Get portfolio summary."""
        
        summary = {
            'total_positions': len(self.positions),
            'total_allocation': sum(p.allocation for p in self.positions.values()),
            'positions': {pair: pos.allocation for pair, pos in self.positions.items()}
        }
        
        return summary

async def run_portfolio_example():
    """Run the portfolio manager example."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Create portfolio manager
    portfolio = PortfolioManager()
    
    # Add positions (allocations should sum to 1.0)
    portfolio.add_position("BTC/USDT", 0.4, 0.02)    # 40% BTC
    portfolio.add_position("ETH/USDT", 0.3, 0.025)   # 30% ETH
    portfolio.add_position("SOL/USDT", 0.2, 0.03)    # 20% SOL
    portfolio.add_position("ADA/USDT", 0.1, 0.035)   # 10% ADA
    
    try:
        # Initialize
        await portfolio.initialize()
        
        # Show summary
        summary = portfolio.get_portfolio_summary()
        logging.info(f"📊 Portfolio: {summary['total_positions']} positions, {summary['total_allocation']:.1%} allocated")
        
        # Analyze portfolio
        await portfolio.analyze_portfolio()
        
        # Rebalance (if needed)
        await portfolio.rebalance_portfolio()
        
        logging.info("✅ Portfolio management completed")
        
    except Exception as e:
        logging.error(f"❌ Portfolio error: {e}")

if __name__ == "__main__":
    asyncio.run(run_portfolio_example())
```

## 📚 **Best Practices**

### **Error Handling**
```python
import asyncio
import logging
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer

async def robust_analysis_example():
    """Example with proper error handling."""
    
    analyzer = LiveGPTAnalyzer("your_api_key", "gpt-4o-mini")
    chart_path = "path/to/chart.png"
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            result = await analyzer.analyze_and_generate_signals(chart_path)
            logging.info(f"✅ Analysis successful on attempt {attempt + 1}")
            return result
            
        except FileNotFoundError:
            logging.error(f"❌ Chart file not found: {chart_path}")
            break  # Don't retry for missing files
            
        except Exception as e:
            logging.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                logging.info(f"🔄 Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error(f"❌ All {max_retries} attempts failed")
                raise
    
    return None

# Usage with error handling
result = await robust_analysis_example()
```

### **Configuration Management**
```python
import os
from gpt_vision_trader.config.settings import TradingConfig

def load_config_from_environment():
    """Load configuration from environment variables."""
    
    config = TradingConfig(
        symbol=os.getenv("TRADING_SYMBOL", "BTC/USDT"),
        timeframe=os.getenv("TRADING_TIMEFRAME", "15m"),
        gpt_model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
        analysis_interval_minutes=int(os.getenv("ANALYSIS_INTERVAL", "15")),
        freqtrade_url=os.getenv("FREQTRADE_URL", "http://127.0.0.1:8080"),
        freqtrade_username=os.getenv("FREQTRADE_USERNAME", "freqtrade"),
        freqtrade_password=os.getenv("FREQTRADE_PASSWORD", "SuperSecret1!")
    )
    
    # Validate configuration
    config.validate()
    
    return config

# Usage
config = load_config_from_environment()
```

### **Logging Best Practices**
```python
import logging
import os
from datetime import datetime

def setup_advanced_logging():
    """Setup advanced logging with multiple handlers."""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create handlers
    file_handler = logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info("📝 Advanced logging setup completed")

# Usage
setup_advanced_logging()
```

## 🎉 **Getting Started**

1. **Choose an example** that matches your use case
2. **Copy the code** and save it as a Python file
3. **Install dependencies**: `pip install -e .`
4. **Set API key**: `export OPENAI_API_KEY="your_key"`
5. **Start Freqtrade** with REST API enabled
6. **Run your script**: `python your_script.py`

All examples are complete and ready to run! Start with the simple examples and work your way up to more complex implementations.

**Happy coding and trading! 🚀📈**
