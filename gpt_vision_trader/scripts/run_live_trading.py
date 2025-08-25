#!/usr/bin/env python3
"""
Live Trading Script
==================

Simplified command-line interface for running live GPT vision trading.
Clean, focused implementation without stake amount calculations.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient, FreqtradeConfig
from gpt_vision_trader.api.trading_controller import TradingController, TradingSession
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader.utils.logging_utils import setup_logging


def create_config_from_args(args) -> TradingConfig:
    """Create trading configuration from command line arguments."""
    if args.profile == 'dev':
        config = TradingConfig.for_development()
    elif args.profile == 'prod':
        config = TradingConfig.for_production()
    else:
        config = TradingConfig.from_run_all_settings()
    
    # Override with command line arguments
    if args.symbol:
        config.symbol = args.symbol
    if args.timeframe:
        config.timeframe = args.timeframe
    if args.gpt_model:
        config.gpt_model = args.gpt_model
    if args.analysis_interval:
        config.analysis_interval_minutes = args.analysis_interval
    if args.freqtrade_url:
        config.freqtrade_url = args.freqtrade_url
    if args.freqtrade_username:
        config.freqtrade_username = args.freqtrade_username
    if args.freqtrade_password:
        config.freqtrade_password = args.freqtrade_password
    
    return config


async def test_connections(config: TradingConfig) -> bool:
    """Test all necessary connections."""
    logger = logging.getLogger(__name__)
    
    # Test OpenAI API key
    if not config.openai_api_key:
        logger.error("❌ OpenAI API key not configured")
        return False
    logger.info("✅ OpenAI API key configured")
    
    # Test Freqtrade API
    freqtrade_config = FreqtradeConfig(
        base_url=config.freqtrade_url,
        username=config.freqtrade_username,
        password=config.freqtrade_password
    )
    
    api_client = FreqtradeAPIClient(freqtrade_config)
    
    if not api_client.ping():
        logger.error("❌ Freqtrade API connection failed")
        logger.error(f"   URL: {config.freqtrade_url}")
        logger.error(f"   Username: {config.freqtrade_username}")
        return False
    
    logger.info("✅ Freqtrade API connection successful")
    
    # Get bot status
    try:
        status = api_client.get_trading_summary()
        logger.info(f"📊 Bot Status: {status['bot_state']}")
        logger.info(f"   Dry Run: {status['dry_run']}")
        logger.info(f"   Open Trades: {status['open_trades_count']}")
        logger.info(f"   Stake Currency: {status['stake_currency']}")
    except Exception as e:
        logger.warning(f"⚠️ Could not get bot status: {e}")
    
    return True


async def run_test_mode(config: TradingConfig) -> bool:
    """Run in test mode - single analysis cycle."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Running in test mode")
    
    try:
        # Initialize components
        freqtrade_config = FreqtradeConfig(
            base_url=config.freqtrade_url,
            username=config.freqtrade_username,
            password=config.freqtrade_password
        )
        
        api_client = FreqtradeAPIClient(freqtrade_config)
        gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
        
        controller = TradingController(
            api_client=api_client,
            gpt_analyzer=gpt_analyzer,
            pair=config.symbol,
            timeframe=config.timeframe,
            visible_days=config.visible_days,
            hidden_days=config.hidden_days,
            indicator_buffer_days=config.indicator_buffer_days
        )
        
        # Run single analysis cycle
        result = await controller.run_analysis_cycle()
        
        if result:
            logger.info("✅ Test analysis cycle completed successfully")
            logger.info(f"   Prediction: {result['prediction']}")
            logger.info(f"   Chart: {result['chart_path']}")
            logger.info(f"   Analysis: {result['analysis'][:100]}...")
            return True
        else:
            logger.error("❌ Test analysis cycle failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test mode failed: {e}")
        return False


async def run_live_trading(config: TradingConfig, max_cycles: int = None) -> None:
    """Run live trading session."""
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting live trading session")
    logger.info(f"   Symbol: {config.symbol}")
    logger.info(f"   Timeframe: {config.timeframe}")
    logger.info(f"   GPT Model: {config.gpt_model}")
    logger.info(f"   Analysis Interval: {config.analysis_interval_minutes} minutes")
    
    try:
        # Initialize components
        freqtrade_config = FreqtradeConfig(
            base_url=config.freqtrade_url,
            username=config.freqtrade_username,
            password=config.freqtrade_password
        )
        
        api_client = FreqtradeAPIClient(freqtrade_config)
        gpt_analyzer = LiveGPTAnalyzer(config.openai_api_key, config.gpt_model)
        
        controller = TradingController(
            api_client=api_client,
            gpt_analyzer=gpt_analyzer,
            pair=config.symbol,
            timeframe=config.timeframe,
            visible_days=config.visible_days,
            hidden_days=config.hidden_days,
            indicator_buffer_days=config.indicator_buffer_days
        )
        
        # Create and run trading session
        session = TradingSession(controller)
        await session.run_session(
            max_cycles=max_cycles,
            cycle_interval_minutes=config.analysis_interval_minutes
        )
        
        # Print final stats
        stats = session.get_session_stats()
        logger.info("📊 Final Session Statistics:")
        logger.info(f"   Cycles completed: {stats['cycles_completed']}")
        logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"   Session duration: {stats['session_duration_seconds']/3600:.1f} hours")
        
    except Exception as e:
        logger.error(f"❌ Live trading session failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GPT Vision Live Trading - Simplified API-based trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (safe)
  python -m gpt_vision_trader.scripts.run_live_trading --test

  # Live trading with default settings
  python -m gpt_vision_trader.scripts.run_live_trading

  # Custom symbol and model
  python -m gpt_vision_trader.scripts.run_live_trading --symbol ETH/USDT --gpt-model gpt-4o-mini

  # Limited cycles for testing
  python -m gpt_vision_trader.scripts.run_live_trading --max-cycles 5 --profile dev
        """
    )
    
    # Configuration profiles
    parser.add_argument('--profile', choices=['default', 'dev', 'prod'], default='default',
                       help='Configuration profile to use')
    
    # Trading parameters
    parser.add_argument('--symbol', type=str, help='Trading pair (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 15m, 1h)')
    parser.add_argument('--gpt-model', type=str, help='GPT model (gpt-4o, gpt-4o-mini)')
    parser.add_argument('--analysis-interval', type=int, help='Analysis interval in minutes')
    
    # Freqtrade API settings
    parser.add_argument('--freqtrade-url', type=str, help='Freqtrade API URL')
    parser.add_argument('--freqtrade-username', type=str, help='Freqtrade API username')
    parser.add_argument('--freqtrade-password', type=str, help='Freqtrade API password')
    
    # Execution settings
    parser.add_argument('--test', action='store_true', help='Run in test mode (single cycle)')
    parser.add_argument('--max-cycles', type=int, help='Maximum number of trading cycles')
    
    # Logging settings
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level, args.log_file)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Validate configuration
        config.validate()
        
        # Log configuration
        logger.info("Configuration:")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            if 'password' not in key.lower():
                logger.info(f"  {key}: {value}")
        
        # Run async main
        if args.test:
            # Test mode
            async def test_main():
                if not await test_connections(config):
                    return False
                return await run_test_mode(config)
            
            success = asyncio.run(test_main())
            sys.exit(0 if success else 1)
        else:
            # Live trading mode
            async def live_main():
                if not await test_connections(config):
                    return
                await run_live_trading(config, args.max_cycles)
            
            asyncio.run(live_main())
            logger.info("🎉 Live trading completed successfully")
    
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
