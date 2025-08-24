#!/usr/bin/env python3
"""
Simple Trading Bot Example
==========================

This script demonstrates how to create a basic trading bot using GPT Vision Trader.

Features:
- Connects to Freqtrade API
- Analyzes charts with GPT vision
- Executes trades automatically
- Comprehensive logging

Usage:
    python examples/simple_trading_bot.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader.api.trading_controller import TradingController


class SimpleTradingBot:
    """
    Simple trading bot that demonstrates basic GPT Vision Trader usage.
    """
    
    def __init__(self, symbol="BTC/USDT", max_cycles=5):
        """
        Initialize the trading bot.
        
        Args:
            symbol: Trading pair to trade
            max_cycles: Maximum number of trading cycles to run
        """
        self.symbol = symbol
        self.max_cycles = max_cycles
        self.running = False
        
        # Use development configuration for safety
        self.config = TradingConfig.for_development()
        self.config.symbol = symbol
        
        # Setup logging
        self._setup_logging()
        
        logging.info(f"🤖 Simple Trading Bot initialized for {symbol}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f"logs/simple_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
    
    async def start(self):
        """Start the trading bot."""
        
        logging.info("🚀 Starting Simple Trading Bot")
        logging.info(f"   Symbol: {self.config.symbol}")
        logging.info(f"   Timeframe: {self.config.timeframe}")
        logging.info(f"   GPT Model: {self.config.gpt_model}")
        logging.info(f"   Max Cycles: {self.max_cycles}")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Run trading loop
            await self._run_trading_loop()
            
        except KeyboardInterrupt:
            logging.info("👋 Bot interrupted by user")
        except Exception as e:
            logging.error(f"❌ Bot error: {e}")
        finally:
            self.stop()
    
    async def _initialize_components(self):
        """Initialize trading components."""
        
        logging.info("🔧 Initializing components...")
        
        # Initialize Freqtrade API client
        freqtrade_config = FreqtradeConfig()
        self.api_client = FreqtradeAPIClient(freqtrade_config)
        
        # Test connection
        if not self.api_client.ping():
            raise ConnectionError("Cannot connect to Freqtrade API. Make sure Freqtrade is running with REST API enabled.")
        
        logging.info("✅ Connected to Freqtrade API")
        
        # Check if we're in dry run mode
        if self.api_client.is_dry_run():
            logging.info("🧪 Freqtrade is in DRY RUN mode (safe for testing)")
        else:
            logging.warning("⚠️ Freqtrade is in LIVE mode - real money at risk!")
        
        # Initialize GPT analyzer
        self.gpt_analyzer = LiveGPTAnalyzer(
            self.config.openai_api_key, 
            self.config.gpt_model
        )
        
        # Create trading controller
        self.controller = TradingController(
            api_client=self.api_client,
            gpt_analyzer=self.gpt_analyzer,
            pair=self.config.symbol,
            timeframe=self.config.timeframe,
            visible_days=self.config.visible_days,
            hidden_days=self.config.hidden_days
        )
        
        logging.info("✅ All components initialized")
    
    async def _run_trading_loop(self):
        """Run the main trading loop."""
        
        logging.info(f"🔄 Starting trading loop ({self.max_cycles} cycles)")
        
        self.running = True
        cycle = 0
        
        while self.running and cycle < self.max_cycles:
            cycle += 1
            
            logging.info(f"\n{'='*50}")
            logging.info(f"🔄 CYCLE {cycle}/{self.max_cycles}")
            logging.info(f"{'='*50}")
            
            try:
                # Run complete trading cycle
                result = await self.controller.run_trading_cycle()
                
                if result:
                    await self._process_cycle_result(cycle, result)
                else:
                    logging.warning(f"⚠️ Cycle {cycle} failed - no result")
                
            except Exception as e:
                logging.error(f"❌ Cycle {cycle} error: {e}")
            
            # Wait before next cycle (unless it's the last one)
            if self.running and cycle < self.max_cycles:
                wait_minutes = self.config.analysis_interval_minutes
                logging.info(f"💤 Waiting {wait_minutes} minutes before next cycle...")
                
                # Sleep in 30-second intervals to allow for graceful shutdown
                for _ in range(wait_minutes * 2):  # 30-second intervals
                    if not self.running:
                        break
                    await asyncio.sleep(30)
        
        logging.info("🏁 Trading loop completed")
    
    async def _process_cycle_result(self, cycle_num, result):
        """Process the result of a trading cycle."""
        
        analysis = result.get('analysis', {})
        execution = result.get('execution', {})
        trading_status = result.get('trading_status', {})
        
        # Log analysis results
        prediction = analysis.get('prediction', 'unknown')
        chart_path = analysis.get('chart_path', 'N/A')
        
        logging.info(f"📊 Analysis Results:")
        logging.info(f"   Prediction: {prediction}")
        logging.info(f"   Chart: {chart_path}")
        logging.info(f"   Analysis: {analysis.get('analysis', 'N/A')[:100]}...")
        
        # Log trading signals
        signals = {k: v for k, v in analysis.items() if k.startswith(('enter_', 'exit_'))}
        active_signals = [k for k, v in signals.items() if v]
        
        if active_signals:
            logging.info(f"📈 Active Signals: {', '.join(active_signals)}")
        else:
            logging.info(f"📊 No active signals")
        
        # Log execution results
        actions_taken = execution.get('actions_taken', [])
        errors = execution.get('errors', [])
        
        if actions_taken:
            logging.info(f"✅ Actions Executed:")
            for action in actions_taken:
                action_type = action.get('action', 'unknown')
                success = action.get('success', False)
                status_icon = "✅" if success else "❌"
                logging.info(f"   {status_icon} {action_type}")
        else:
            logging.info(f"ℹ️ No actions taken")
        
        if errors:
            logging.warning(f"⚠️ Execution Errors:")
            for error in errors:
                logging.warning(f"   {error}")
        
        # Log current trading status
        bot_state = trading_status.get('bot_state', 'unknown')
        open_trades = trading_status.get('open_trades_count', 0)
        total_profit = trading_status.get('total_profit', 0)
        
        logging.info(f"💼 Trading Status:")
        logging.info(f"   Bot State: {bot_state}")
        logging.info(f"   Open Trades: {open_trades}")
        logging.info(f"   Total Profit: {total_profit:.4f}")
        
        # Show cycle summary
        logging.info(f"📋 Cycle {cycle_num} Summary: {prediction} → {len(actions_taken)} actions")
    
    def stop(self):
        """Stop the trading bot."""
        
        logging.info("🛑 Stopping trading bot...")
        self.running = False
        
        if hasattr(self, 'controller'):
            try:
                # Get final statistics
                stats = self.controller.get_status()
                controller_stats = stats.get('controller_info', {})
                
                logging.info("📊 Final Statistics:")
                logging.info(f"   Total Cycles: {controller_stats.get('cycle_count', 0)}")
                logging.info(f"   Last Prediction: {controller_stats.get('last_prediction', 'N/A')}")
                
            except Exception as e:
                logging.warning(f"⚠️ Could not get final stats: {e}")
        
        logging.info("✅ Trading bot stopped")


async def main():
    """Main function to run the simple trading bot."""
    
    print("🤖 Simple Trading Bot")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your_api_key_here'")
        return
    
    print("✅ OpenAI API key found")
    
    # Configuration
    symbol = "BTC/USDT"  # You can change this
    max_cycles = 3       # Run 3 cycles for demo
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Max Cycles: {max_cycles}")
    print(f"   Model: gpt-4o-mini (development)")
    
    # Create and start bot
    bot = SimpleTradingBot(symbol=symbol, max_cycles=max_cycles)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        bot.stop()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Freqtrade is running with REST API enabled")
        print("2. Check your OpenAI API key")
        print("3. Verify network connectivity")


if __name__ == "__main__":
    asyncio.run(main())
