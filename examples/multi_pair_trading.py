#!/usr/bin/env python3
"""
Multi-Pair Trading Example
==========================

This script demonstrates how to trade multiple cryptocurrency pairs
simultaneously with individual configurations and risk management.

Features:
- Trade multiple pairs concurrently
- Individual configurations per pair
- Cost optimization with different GPT models
- Portfolio-level risk management
- Comprehensive monitoring

Usage:
    python examples/multi_pair_trading.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader.api.trading_controller import TradingController


@dataclass
class PairConfig:
    """Configuration for a trading pair."""
    symbol: str
    timeframe: str
    gpt_model: str
    analysis_interval: int
    priority: int = 1  # 1=high, 2=medium, 3=low
    max_risk: float = 0.02  # 2% max risk per trade


class MultiPairTradingSystem:
    """
    Multi-pair trading system that manages multiple cryptocurrency pairs
    with individual configurations and risk management.
    """
    
    def __init__(self):
        """Initialize the multi-pair trading system."""
        
        self.pairs: Dict[str, PairConfig] = {}
        self.controllers: Dict[str, TradingController] = {}
        self.api_client = None
        self.running = False
        
        # Portfolio settings
        self.max_concurrent_trades = 3
        self.total_portfolio_risk = 0.10  # 10% max portfolio risk
        
        # Setup logging
        self._setup_logging()
        
        logging.info("🚀 Multi-Pair Trading System initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/multi_pair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
    
    def add_pair(self, symbol: str, timeframe: str = "15m", gpt_model: str = "gpt-4o-mini", 
                 analysis_interval: int = 15, priority: int = 1, max_risk: float = 0.02):
        """
        Add a trading pair to the system.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Chart timeframe
            gpt_model: GPT model to use
            analysis_interval: Minutes between analyses
            priority: Priority level (1=high, 2=medium, 3=low)
            max_risk: Maximum risk per trade
        """
        
        pair_config = PairConfig(
            symbol=symbol,
            timeframe=timeframe,
            gpt_model=gpt_model,
            analysis_interval=analysis_interval,
            priority=priority,
            max_risk=max_risk
        )
        
        self.pairs[symbol] = pair_config
        
        logging.info(f"➕ Added {symbol}: {timeframe}, {gpt_model}, {analysis_interval}min, priority={priority}")
    
    async def initialize(self):
        """Initialize the trading system."""
        
        logging.info("🔧 Initializing multi-pair trading system...")
        
        # Initialize Freqtrade API client
        freqtrade_config = FreqtradeConfig()
        self.api_client = FreqtradeAPIClient(freqtrade_config)
        
        # Test connection
        if not self.api_client.ping():
            raise ConnectionError("Cannot connect to Freqtrade API")
        
        logging.info("✅ Connected to Freqtrade API")
        
        # Check trading mode
        if self.api_client.is_dry_run():
            logging.info("🧪 Freqtrade is in DRY RUN mode")
        else:
            logging.warning("⚠️ Freqtrade is in LIVE mode - real money at risk!")
        
        # Initialize controllers for each pair
        for symbol, pair_config in self.pairs.items():
            await self._initialize_pair_controller(symbol, pair_config)
        
        logging.info(f"✅ Initialized {len(self.controllers)} pair controllers")
    
    async def _initialize_pair_controller(self, symbol: str, pair_config: PairConfig):
        """Initialize controller for a specific pair."""
        
        try:
            # Create configuration for this pair
            config = TradingConfig(
                symbol=symbol,
                timeframe=pair_config.timeframe,
                gpt_model=pair_config.gpt_model,
                analysis_interval_minutes=pair_config.analysis_interval
            )
            
            # Create GPT analyzer
            gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
            
            # Create trading controller
            controller = TradingController(
                api_client=self.api_client,
                gpt_analyzer=gpt_analyzer,
                pair=symbol,
                timeframe=pair_config.timeframe,
                visible_days=config.visible_days,
                hidden_days=config.hidden_days,
                indicator_buffer_days=config.indicator_buffer_days
            )
            
            self.controllers[symbol] = controller
            
            logging.info(f"✅ Initialized controller for {symbol}")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize {symbol}: {e}")
    
    async def run_analysis_round(self):
        """Run one analysis round for all pairs."""
        
        logging.info("\n" + "="*60)
        logging.info("🔄 Starting Analysis Round")
        logging.info("="*60)
        
        # Get current portfolio status
        portfolio_status = await self._get_portfolio_status()
        
        # Analyze pairs by priority
        pairs_by_priority = sorted(
            self.pairs.items(),
            key=lambda x: x[1].priority
        )
        
        results = {}
        
        for symbol, pair_config in pairs_by_priority:
            if symbol not in self.controllers:
                logging.warning(f"⚠️ No controller for {symbol}")
                continue
            
            logging.info(f"\n📊 Analyzing {symbol} (Priority {pair_config.priority})")
            
            try:
                # Run analysis
                controller = self.controllers[symbol]
                result = await controller.run_analysis_cycle()
                
                if result:
                    prediction = result.get('prediction', 'unknown')
                    signals = {k: v for k, v in result.items() if k.startswith(('enter_', 'exit_'))}
                    active_signals = [k for k, v in signals.items() if v]
                    
                    logging.info(f"   🤖 Prediction: {prediction}")
                    logging.info(f"   📈 Active Signals: {active_signals if active_signals else 'None'}")
                    
                    results[symbol] = result
                else:
                    logging.warning(f"   ❌ Analysis failed for {symbol}")
                    results[symbol] = None
                
            except Exception as e:
                logging.error(f"   ❌ Error analyzing {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    async def execute_trading_decisions(self, analysis_results: Dict):
        """Execute trading decisions based on analysis results."""
        
        logging.info("\n💼 Executing Trading Decisions")
        logging.info("-" * 40)
        
        # Get current open trades
        open_trades = self.api_client.get_open_trades()
        current_pairs = {trade.get('pair') for trade in open_trades}
        
        logging.info(f"📊 Current open trades: {len(open_trades)}/{self.max_concurrent_trades}")
        
        execution_results = {}
        
        for symbol, result in analysis_results.items():
            if not result:
                continue
            
            logging.info(f"\n🔄 Processing {symbol}:")
            
            try:
                # Check if we can open new positions
                can_open_new = len(open_trades) < self.max_concurrent_trades
                has_open_position = symbol in current_pairs
                
                if not can_open_new and not has_open_position:
                    logging.info(f"   ⏸️ Max concurrent trades reached, skipping new entry")
                    continue
                
                # Execute trading signals
                controller = self.controllers[symbol]
                execution = await controller.execute_trading_signals(result)
                
                actions_taken = execution.get('actions_taken', [])
                errors = execution.get('errors', [])
                
                if actions_taken:
                    logging.info(f"   ✅ Actions executed:")
                    for action in actions_taken:
                        action_type = action.get('action', 'unknown')
                        success = action.get('success', False)
                        status_icon = "✅" if success else "❌"
                        logging.info(f"      {status_icon} {action_type}")
                else:
                    logging.info(f"   ℹ️ No actions taken")
                
                if errors:
                    logging.warning(f"   ⚠️ Errors: {len(errors)}")
                    for error in errors:
                        logging.warning(f"      {error}")
                
                execution_results[symbol] = execution
                
            except Exception as e:
                logging.error(f"   ❌ Execution error for {symbol}: {e}")
                execution_results[symbol] = {'error': str(e)}
        
        return execution_results
    
    async def _get_portfolio_status(self):
        """Get current portfolio status."""
        
        try:
            summary = self.api_client.get_trading_summary()
            
            status = {
                'total_profit': summary.get('total_profit', 0),
                'open_trades': summary.get('open_trades_count', 0),
                'max_trades': summary.get('max_open_trades', 0),
                'win_rate': summary.get('win_rate', 0),
                'bot_state': summary.get('bot_state', 'unknown')
            }
            
            logging.info(f"💰 Portfolio Status:")
            logging.info(f"   Total Profit: {status['total_profit']:.4f}")
            logging.info(f"   Open Trades: {status['open_trades']}/{status['max_trades']}")
            logging.info(f"   Win Rate: {status['win_rate']:.1f}%")
            logging.info(f"   Bot State: {status['bot_state']}")
            
            return status
            
        except Exception as e:
            logging.error(f"❌ Error getting portfolio status: {e}")
            return {}
    
    async def run_session(self, max_rounds: int = 5, round_interval: int = 30):
        """
        Run a complete multi-pair trading session.
        
        Args:
            max_rounds: Maximum number of analysis rounds
            round_interval: Minutes between rounds
        """
        
        logging.info(f"🚀 Starting multi-pair session ({max_rounds} rounds)")
        
        self.running = True
        round_num = 0
        
        try:
            while self.running and round_num < max_rounds:
                round_num += 1
                
                logging.info(f"\n🔄 ROUND {round_num}/{max_rounds}")
                
                # Run analysis for all pairs
                analysis_results = await self.run_analysis_round()
                
                # Execute trading decisions
                execution_results = await self.execute_trading_decisions(analysis_results)
                
                # Show round summary
                successful_analyses = sum(1 for r in analysis_results.values() if r is not None)
                total_actions = sum(
                    len(r.get('actions_taken', [])) 
                    for r in execution_results.values() 
                    if isinstance(r, dict) and 'actions_taken' in r
                )
                
                logging.info(f"\n📋 Round {round_num} Summary:")
                logging.info(f"   Successful Analyses: {successful_analyses}/{len(self.pairs)}")
                logging.info(f"   Total Actions: {total_actions}")
                
                # Wait before next round
                if self.running and round_num < max_rounds:
                    logging.info(f"💤 Waiting {round_interval} minutes before next round...")
                    
                    for _ in range(round_interval * 2):  # 30-second intervals
                        if not self.running:
                            break
                        await asyncio.sleep(30)
            
            logging.info("🏁 Multi-pair session completed")
            
        except KeyboardInterrupt:
            logging.info("👋 Session interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the multi-pair trading system."""
        
        logging.info("🛑 Stopping multi-pair trading system...")
        self.running = False
        
        # Get final statistics
        try:
            final_status = asyncio.create_task(self._get_portfolio_status())
            logging.info("📊 Final portfolio status logged")
        except:
            pass
        
        logging.info("✅ Multi-pair system stopped")


async def main():
    """Main function to run the multi-pair trading example."""
    
    print("🚀 Multi-Pair Trading System")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    print("✅ OpenAI API key found")
    
    # Create trading system
    system = MultiPairTradingSystem()
    
    # Add trading pairs with different configurations
    print("\n📊 Adding trading pairs:")
    
    # High priority pairs (better models, frequent analysis)
    system.add_pair("BTC/USDT", "15m", "gpt-4o-mini", 15, priority=1, max_risk=0.02)
    system.add_pair("ETH/USDT", "15m", "gpt-4o-mini", 15, priority=1, max_risk=0.025)
    
    # Medium priority pairs (cheaper model, less frequent)
    system.add_pair("SOL/USDT", "30m", "gpt-4o-mini", 30, priority=2, max_risk=0.03)
    system.add_pair("ADA/USDT", "30m", "gpt-4o-mini", 30, priority=2, max_risk=0.035)
    
    # Low priority pairs (experimental)
    system.add_pair("MATIC/USDT", "1h", "gpt-4o-mini", 60, priority=3, max_risk=0.04)
    
    print(f"✅ Added {len(system.pairs)} trading pairs")
    
    try:
        # Initialize system
        print("\n🔧 Initializing system...")
        await system.initialize()
        
        # Run trading session
        print("\n🔄 Starting trading session...")
        await system.run_session(max_rounds=3, round_interval=5)  # Short demo
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        system.stop()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Freqtrade is running with REST API enabled")
        print("2. Check your OpenAI API key")
        print("3. Verify all trading pairs are available on your exchange")


if __name__ == "__main__":
    asyncio.run(main())
