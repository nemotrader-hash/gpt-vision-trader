#!/usr/bin/env python3
"""
Portfolio Manager Example
=========================

This script demonstrates advanced portfolio management with:
- Position sizing based on portfolio allocation
- Risk management across multiple pairs
- Performance tracking and rebalancing
- Real-time monitoring and alerts

Features:
- Portfolio-based position sizing
- Risk management per position and total portfolio
- Performance analytics
- Automated rebalancing
- Real-time monitoring

Usage:
    python examples/portfolio_manager.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer


@dataclass
class PortfolioPosition:
    """Portfolio position configuration."""
    symbol: str
    target_allocation: float  # Target portfolio allocation (0.0 to 1.0)
    max_position_risk: float  # Max risk per position (e.g., 0.02 = 2%)
    min_position_size: float  # Minimum position size in USD
    max_position_size: float  # Maximum position size in USD
    priority: int = 1  # Trading priority (1=high, 3=low)
    
    # Current status
    current_allocation: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class PortfolioManager:
    """
    Advanced portfolio manager that handles multiple cryptocurrency positions
    with sophisticated risk management and performance tracking.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Initial portfolio value in USD
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        
        # Risk management settings
        self.max_portfolio_risk = 0.10  # 10% max total portfolio risk
        self.max_single_position = 0.30  # 30% max single position
        self.rebalance_threshold = 0.05  # 5% allocation deviation triggers rebalance
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        
        # API client
        self.api_client = None
        
        # Setup logging
        self._setup_logging()
        
        logging.info(f"💼 Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f"logs/portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
    
    def add_position(self, symbol: str, target_allocation: float, 
                    max_position_risk: float = 0.02, priority: int = 1):
        """
        Add a position to the portfolio.
        
        Args:
            symbol: Trading pair symbol
            target_allocation: Target portfolio allocation (0.0 to 1.0)
            max_position_risk: Maximum risk per position
            priority: Trading priority (1=high, 3=low)
        """
        
        # Calculate position size limits
        target_value = self.initial_capital * target_allocation
        min_size = max(target_value * 0.1, 100)  # At least 10% of target or $100
        max_size = target_value * 2.0  # Max 200% of target
        
        position = PortfolioPosition(
            symbol=symbol,
            target_allocation=target_allocation,
            max_position_risk=max_position_risk,
            min_position_size=min_size,
            max_position_size=max_size,
            priority=priority
        )
        
        self.positions[symbol] = position
        
        logging.info(f"➕ Added {symbol}: {target_allocation:.1%} allocation, "
                    f"${min_size:.0f}-${max_size:.0f} size range")
    
    async def initialize(self):
        """Initialize the portfolio manager."""
        
        logging.info("🔧 Initializing Portfolio Manager...")
        
        # Initialize Freqtrade API
        freqtrade_config = FreqtradeConfig()
        self.api_client = FreqtradeAPIClient(freqtrade_config)
        
        if not self.api_client.ping():
            raise ConnectionError("Cannot connect to Freqtrade API")
        
        logging.info("✅ Connected to Freqtrade API")
        
        # Update current portfolio status
        await self.update_portfolio_status()
        
        logging.info("✅ Portfolio Manager initialized")
    
    async def update_portfolio_status(self):
        """Update current portfolio status from Freqtrade."""
        
        try:
            # Get trading summary
            summary = self.api_client.get_trading_summary()
            
            # Update current capital
            total_profit = summary.get('total_profit', 0)
            self.current_capital = self.initial_capital + total_profit
            
            # Get open trades
            open_trades = self.api_client.get_open_trades()
            
            # Update position status
            for symbol, position in self.positions.items():
                # Find open trade for this symbol
                open_trade = next((t for t in open_trades if t.get('pair') == symbol), None)
                
                if open_trade:
                    # Position is open
                    stake_amount = open_trade.get('stake_amount', 0)
                    profit_ratio = open_trade.get('profit_ratio', 0)
                    
                    position.current_value = stake_amount
                    position.current_allocation = stake_amount / self.current_capital
                    position.unrealized_pnl = stake_amount * profit_ratio
                else:
                    # No open position
                    position.current_value = 0
                    position.current_allocation = 0
                    position.unrealized_pnl = 0
            
            logging.info(f"📊 Portfolio updated: ${self.current_capital:,.2f} total value")
            
        except Exception as e:
            logging.error(f"❌ Error updating portfolio status: {e}")
    
    async def analyze_portfolio(self):
        """Analyze all positions in the portfolio."""
        
        logging.info("\n" + "="*60)
        logging.info("📊 Portfolio Analysis")
        logging.info("="*60)
        
        await self.update_portfolio_status()
        
        # Portfolio overview
        total_allocated = sum(pos.current_allocation for pos in self.positions.values())
        total_target = sum(pos.target_allocation for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        logging.info(f"💰 Portfolio Overview:")
        logging.info(f"   Current Value: ${self.current_capital:,.2f}")
        logging.info(f"   Total P&L: ${total_pnl:+,.2f} ({total_pnl/self.initial_capital:+.2%})")
        logging.info(f"   Allocated: {total_allocated:.1%} (Target: {total_target:.1%})")
        
        # Analyze individual positions
        logging.info(f"\n📈 Position Analysis:")
        
        analysis_results = {}
        
        for symbol, position in self.positions.items():
            logging.info(f"\n🔍 {symbol}:")
            logging.info(f"   Target: {position.target_allocation:.1%} | "
                        f"Current: {position.current_allocation:.1%}")
            logging.info(f"   Value: ${position.current_value:,.2f} | "
                        f"P&L: ${position.unrealized_pnl:+,.2f}")
            
            # Calculate allocation deviation
            allocation_deviation = abs(position.current_allocation - position.target_allocation)
            needs_rebalancing = allocation_deviation > self.rebalance_threshold
            
            if needs_rebalancing:
                logging.info(f"   ⚠️ Needs rebalancing: {allocation_deviation:.1%} deviation")
            else:
                logging.info(f"   ✅ Allocation OK: {allocation_deviation:.1%} deviation")
            
            # Get market data
            try:
                df = self.api_client.get_pair_candles(symbol, "15m", limit=50)
                
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    price_change_24h = ((df['close'].iloc[-1] / df['close'].iloc[-25]) - 1) * 100
                    
                    logging.info(f"   💱 Price: ${current_price:.4f} ({price_change_24h:+.2f}% 24h)")
                    
                    analysis_results[symbol] = {
                        'needs_rebalancing': needs_rebalancing,
                        'allocation_deviation': allocation_deviation,
                        'current_price': current_price,
                        'price_change_24h': price_change_24h,
                        'position': position
                    }
                else:
                    logging.warning(f"   ⚠️ No market data available")
                    
            except Exception as e:
                logging.error(f"   ❌ Error getting market data: {e}")
        
        return analysis_results
    
    async def generate_trading_signals(self):
        """Generate trading signals based on portfolio analysis and GPT predictions."""
        
        logging.info("\n🤖 Generating Trading Signals")
        logging.info("-" * 40)
        
        signals = {}
        
        for symbol, position in self.positions.items():
            logging.info(f"\n📊 Analyzing {symbol}:")
            
            try:
                # Create GPT analyzer for this symbol
                config = TradingConfig(
                    symbol=symbol,
                    gpt_model="gpt-4o-mini"  # Use cheaper model for portfolio
                )
                
                analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
                
                # Note: In a real implementation, you would generate a chart here
                # For this example, we'll simulate signals based on allocation deviation
                
                allocation_deviation = abs(position.current_allocation - position.target_allocation)
                
                if allocation_deviation > self.rebalance_threshold:
                    if position.current_allocation < position.target_allocation:
                        # Under-allocated - consider buying
                        signals[symbol] = {
                            'action': 'buy',
                            'reason': 'under_allocated',
                            'target_size': position.target_allocation * self.current_capital,
                            'current_size': position.current_value,
                            'deviation': allocation_deviation
                        }
                        logging.info(f"   📈 BUY signal: Under-allocated by {allocation_deviation:.1%}")
                    else:
                        # Over-allocated - consider selling
                        signals[symbol] = {
                            'action': 'sell',
                            'reason': 'over_allocated',
                            'target_size': position.target_allocation * self.current_capital,
                            'current_size': position.current_value,
                            'deviation': allocation_deviation
                        }
                        logging.info(f"   📉 SELL signal: Over-allocated by {allocation_deviation:.1%}")
                else:
                    signals[symbol] = {
                        'action': 'hold',
                        'reason': 'allocation_ok',
                        'deviation': allocation_deviation
                    }
                    logging.info(f"   ✅ HOLD: Allocation within threshold")
                
            except Exception as e:
                logging.error(f"   ❌ Error analyzing {symbol}: {e}")
                signals[symbol] = {'action': 'error', 'reason': str(e)}
        
        return signals
    
    async def execute_rebalancing(self, signals: Dict):
        """Execute portfolio rebalancing based on signals."""
        
        logging.info("\n⚖️ Executing Portfolio Rebalancing")
        logging.info("-" * 40)
        
        execution_results = {}
        
        # Sort signals by priority and action (sells first, then buys)
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: (
                self.positions[x[0]].priority,
                0 if x[1].get('action') == 'sell' else 1  # Sells first
            )
        )
        
        for symbol, signal in sorted_signals:
            if signal.get('action') in ['buy', 'sell']:
                logging.info(f"\n🔄 Processing {signal['action'].upper()} for {symbol}:")
                
                try:
                    target_size = signal.get('target_size', 0)
                    current_size = signal.get('current_size', 0)
                    size_diff = target_size - current_size
                    
                    logging.info(f"   Target: ${target_size:.2f} | Current: ${current_size:.2f}")
                    logging.info(f"   Difference: ${size_diff:+.2f}")
                    
                    if abs(size_diff) < self.positions[symbol].min_position_size:
                        logging.info(f"   ⏸️ Size difference too small to trade")
                        execution_results[symbol] = {'action': 'skipped', 'reason': 'size_too_small'}
                        continue
                    
                    # In a real implementation, you would execute the trade here
                    # For this example, we'll just log what would happen
                    
                    if signal['action'] == 'buy':
                        logging.info(f"   ✅ Would BUY ${abs(size_diff):.2f} worth of {symbol}")
                        # self.api_client.force_enter(symbol, 'long', entry_tag='portfolio_rebalance')
                    else:
                        logging.info(f"   ✅ Would SELL ${abs(size_diff):.2f} worth of {symbol}")
                        # self.api_client.force_exit(symbol)
                    
                    execution_results[symbol] = {
                        'action': signal['action'],
                        'size': abs(size_diff),
                        'executed': False,  # Set to True when actually executing
                        'reason': signal['reason']
                    }
                    
                except Exception as e:
                    logging.error(f"   ❌ Execution error: {e}")
                    execution_results[symbol] = {'action': 'error', 'reason': str(e)}
            else:
                execution_results[symbol] = {'action': signal.get('action', 'none')}
        
        return execution_results
    
    async def run_portfolio_cycle(self):
        """Run one complete portfolio management cycle."""
        
        logging.info("\n" + "="*80)
        logging.info(f"🔄 Portfolio Management Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*80)
        
        try:
            # 1. Analyze portfolio
            analysis = await self.analyze_portfolio()
            
            # 2. Generate trading signals
            signals = await self.generate_trading_signals()
            
            # 3. Execute rebalancing
            execution_results = await self.execute_rebalancing(signals)
            
            # 4. Update performance tracking
            await self._update_performance_tracking()
            
            # 5. Show cycle summary
            total_signals = len([s for s in signals.values() if s.get('action') in ['buy', 'sell']])
            total_executed = len([r for r in execution_results.values() if r.get('executed')])
            
            logging.info(f"\n📋 Cycle Summary:")
            logging.info(f"   Positions Analyzed: {len(analysis)}")
            logging.info(f"   Trading Signals: {total_signals}")
            logging.info(f"   Trades Executed: {total_executed}")
            
            return {
                'analysis': analysis,
                'signals': signals,
                'execution_results': execution_results
            }
            
        except Exception as e:
            logging.error(f"❌ Portfolio cycle error: {e}")
            return None
    
    async def _update_performance_tracking(self):
        """Update performance tracking data."""
        
        try:
            await self.update_portfolio_status()
            
            # Calculate portfolio metrics
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            total_allocated = sum(pos.current_allocation for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            performance_data = {
                'timestamp': datetime.now(),
                'portfolio_value': self.current_capital,
                'total_return': total_return,
                'total_allocated': total_allocated,
                'unrealized_pnl': total_unrealized_pnl,
                'position_count': len([p for p in self.positions.values() if p.current_value > 0])
            }
            
            self.performance_history.append(performance_data)
            
            # Keep only last 100 records
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
        except Exception as e:
            logging.error(f"❌ Error updating performance tracking: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'positions': {},
            'risk_metrics': {}
        }
        
        # Position summaries
        for symbol, position in self.positions.items():
            summary['positions'][symbol] = {
                'target_allocation': position.target_allocation,
                'current_allocation': position.current_allocation,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl,
                'deviation': abs(position.current_allocation - position.target_allocation)
            }
        
        # Risk metrics
        total_risk = sum(pos.current_allocation * pos.max_position_risk for pos in self.positions.values())
        max_position_size = max((pos.current_allocation for pos in self.positions.values()), default=0)
        
        summary['risk_metrics'] = {
            'total_portfolio_risk': total_risk,
            'max_position_risk': self.max_portfolio_risk,
            'largest_position': max_position_size,
            'max_single_position': self.max_single_position
        }
        
        return summary


async def main():
    """Main function to run the portfolio manager example."""
    
    print("💼 Portfolio Manager Example")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    print("✅ OpenAI API key found")
    
    # Create portfolio manager
    portfolio = PortfolioManager(initial_capital=10000.0)
    
    # Define portfolio allocation
    print("\n📊 Setting up portfolio allocation:")
    
    portfolio.add_position("BTC/USDT", 0.35, 0.02, priority=1)  # 35% BTC
    portfolio.add_position("ETH/USDT", 0.25, 0.025, priority=1)  # 25% ETH
    portfolio.add_position("SOL/USDT", 0.15, 0.03, priority=2)   # 15% SOL
    portfolio.add_position("ADA/USDT", 0.10, 0.035, priority=2)  # 10% ADA
    portfolio.add_position("MATIC/USDT", 0.10, 0.04, priority=3) # 10% MATIC
    portfolio.add_position("DOT/USDT", 0.05, 0.045, priority=3)  # 5% DOT
    
    total_allocation = sum(pos.target_allocation for pos in portfolio.positions.values())
    print(f"✅ Portfolio setup complete: {total_allocation:.1%} allocated across {len(portfolio.positions)} positions")
    
    try:
        # Initialize portfolio manager
        print("\n🔧 Initializing portfolio manager...")
        await portfolio.initialize()
        
        # Run portfolio management cycles
        print("\n🔄 Running portfolio management cycles...")
        
        for cycle in range(3):  # Run 3 cycles for demo
            print(f"\n📊 Cycle {cycle + 1}/3")
            
            result = await portfolio.run_portfolio_cycle()
            
            if result:
                print("✅ Cycle completed successfully")
            else:
                print("❌ Cycle failed")
            
            # Wait between cycles (shortened for demo)
            if cycle < 2:
                print("💤 Waiting 2 minutes before next cycle...")
                await asyncio.sleep(120)
        
        # Show final portfolio summary
        print("\n📊 Final Portfolio Summary:")
        summary = portfolio.get_portfolio_summary()
        
        print(f"💰 Portfolio Value: ${summary['current_capital']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:+.2%}")
        print(f"⚠️ Total Risk: {summary['risk_metrics']['total_portfolio_risk']:.1%}")
        
        print("\n🎉 Portfolio management demo completed!")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Freqtrade is running with REST API enabled")
        print("2. Check your OpenAI API key")
        print("3. Verify all trading pairs are available")


if __name__ == "__main__":
    asyncio.run(main())
