#!/usr/bin/env python3
"""
Trading Controller
=================

Main trading controller that orchestrates GPT analysis and trade execution.
Simplified version focused on core functionality without complex position sizing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from ..core.gpt_analyzer import LiveGPTAnalyzer
from ..core.chart_generator import ChartGenerator
from ..core.data_processor import DataProcessor
from ..utils.logging_utils import TradingLogHandler
from .freqtrade_client import FreqtradeAPIClient, TradingOperations
from ..utils.indicators import SMAIndicator


class TradingController:
    """
    Main trading controller that coordinates all trading activities.
    
    Simplified workflow:
    1. Get live data from Freqtrade API
    2. Generate chart with technical indicators  
    3. Analyze chart with GPT
    4. Execute trades via forceenter/forceexit
    """
    
    def __init__(self,
                 api_client: FreqtradeAPIClient,
                 gpt_analyzer: LiveGPTAnalyzer,
                 pair: str,
                 timeframe: str,
                 visible_days: int = 6,
                 hidden_days: int = 1,
                 technical_indicators: Optional[Dict[str, SMAIndicator]] = None):
        """
        Initialize trading controller.
        
        Args:
            api_client: Freqtrade API client
            gpt_analyzer: GPT analyzer instance
            pair: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '15m')
            visible_days: Days of visible data for analysis
            hidden_days: Days of hidden/future data (prediction horizon)
        """
        self.api_client = api_client
        self.gpt_analyzer = gpt_analyzer
        self.pair = pair
        self.timeframe = timeframe
        
        # Set up default technical indicators if none provided
        if technical_indicators is None:
            technical_indicators = {
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
        
        # Initialize components
        self.data_processor = DataProcessor(visible_days, hidden_days)
        self.chart_generator = ChartGenerator(
            "gpt_vision_trader/data/temp_charts",
            technical_indicators=technical_indicators
        )
        self.trading_ops = TradingOperations(api_client)
        self.trading_logger = TradingLogHandler()
        
        # State tracking
        self.cycle_count = 0
        self.last_analysis_time: Optional[datetime] = None
        self.last_prediction: Optional[str] = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TradingController initialized for {pair} {timeframe}")
    
    async def run_analysis_cycle(self) -> Optional[Dict]:
        """
        Run a complete analysis cycle.
        
        Returns:
            Analysis results dictionary or None if failed
        """
        try:
            self.cycle_count += 1
            self.logger.info(f"Starting analysis cycle #{self.cycle_count}")
            
            # Step 1: Get live OHLCV data
            ohlcv_data = self.api_client.get_pair_candles(
                self.pair, 
                self.timeframe, 
                limit=500
            )
            
            if ohlcv_data.empty:
                self.logger.error("No OHLCV data received")
                return None
            
            # Step 2: Extract analysis window
            visible_data, hidden_placeholder = self.data_processor.extract_analysis_window(
                ohlcv_data, self.timeframe
            )
            
            # Step 3: Generate chart
            chart_path = self.chart_generator.generate_live_chart(
                visible_data,
                hidden_placeholder,
                f"{self.pair} {self.timeframe} - Live Analysis",
                self.pair,
                self.timeframe
            )
            
            # Step 4: Analyze with GPT
            analysis_result = await self.gpt_analyzer.analyze_and_generate_signals(chart_path)
            
            # Step 5: Prepare results
            self.last_analysis_time = datetime.now()
            self.last_prediction = analysis_result['prediction']
            
            # Step 6: Log analysis
            self.trading_logger.log_analysis(
                chart_path,
                analysis_result['prediction'],
                analysis_result['analysis'],
                self.pair,
                self.timeframe
            )
            
            result = {
                'cycle_id': self.cycle_count,
                'timestamp': self.last_analysis_time.isoformat(),
                'pair': self.pair,
                'timeframe': self.timeframe,
                'chart_path': chart_path,
                'ohlcv_data_points': len(ohlcv_data),
                'visible_data_points': len(visible_data),
                **analysis_result
            }
            
            self.logger.info(f"Analysis cycle #{self.cycle_count} completed: {self.last_prediction}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in analysis cycle #{self.cycle_count}: {e}")
            return None
    
    async def execute_trading_signals(self, analysis_result: Dict) -> Dict:
        """
        Execute trading signals based on analysis results.
        
        Args:
            analysis_result: Results from analysis cycle
            
        Returns:
            Execution results dictionary
        """
        try:
            # Extract signals
            signals = {
                'enter_long': analysis_result.get('enter_long', False),
                'exit_long': analysis_result.get('exit_long', False),
                'enter_short': analysis_result.get('enter_short', False),
                'exit_short': analysis_result.get('exit_short', False)
            }
            
            # Log signals
            self.trading_logger.log_trade_signal(
                signals,
                self.pair,
                analysis_result['prediction']
            )
            
            # Execute signals with GPT reasoning included
            execution_results = self.trading_ops.execute_trading_signals_with_reasoning(
                signals, self.pair, analysis_result
            )
            
            # Log execution results
            for action, success in execution_results.items():
                if action.endswith('_success') and success:
                    base_action = action.replace('_success', '')
                    self.trading_logger.log_trade_execution(
                        base_action,
                        self.pair,
                        success
                    )
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'signals': signals,
                'execution_results': execution_results,
                'any_action_taken': any(execution_results.values())
            }
            
            self.logger.info(f"Signal execution completed: {result['any_action_taken']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trading signals: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'any_action_taken': False
            }
    
    async def run_trading_cycle(self) -> Optional[Dict]:
        """
        Run a complete trading cycle: analyze + execute.
        
        Returns:
            Complete cycle results or None if failed
        """
        try:
            # Run analysis
            analysis_result = await self.run_analysis_cycle()
            if not analysis_result:
                return None
            
            # Execute trading signals
            execution_result = await self.execute_trading_signals(analysis_result)
            
            # Get current trading status
            trading_summary = self.api_client.get_trading_summary()
            
            # Combine results
            cycle_result = {
                'cycle_id': self.cycle_count,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis_result,
                'execution': execution_result,
                'trading_status': trading_summary
            }
            
            return cycle_result
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            return None
    
    def get_status(self) -> Dict:
        """
        Get current controller status.
        
        Returns:
            Status dictionary
        """
        return {
            'controller_info': {
                'pair': self.pair,
                'timeframe': self.timeframe,
                'cycle_count': self.cycle_count,
                'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                'last_prediction': self.last_prediction
            },
            'api_status': self.api_client.get_trading_summary(),
            'gpt_stats': self.gpt_analyzer.get_stats()
        }


class TradingSession:
    """
    Manages a complete trading session with multiple cycles.
    """
    
    def __init__(self, controller: TradingController):
        """
        Initialize trading session.
        
        Args:
            controller: Trading controller instance
        """
        self.controller = controller
        self.session_start = datetime.now()
        self.cycles_completed = 0
        self.cycles_successful = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trading session initialized")
    
    async def run_session(self,
                         max_cycles: Optional[int] = None,
                         cycle_interval_minutes: int = 15) -> None:
        """
        Run a trading session with multiple cycles.
        
        Args:
            max_cycles: Maximum number of cycles (None for unlimited)
            cycle_interval_minutes: Minutes between cycles
        """
        self.logger.info(f"Starting trading session - Max cycles: {max_cycles}, Interval: {cycle_interval_minutes}m")
        
        try:
            while max_cycles is None or self.cycles_completed < max_cycles:
                cycle_start = datetime.now()
                
                # Run trading cycle
                result = await self.controller.run_trading_cycle()
                
                self.cycles_completed += 1
                if result:
                    self.cycles_successful += 1
                    self.logger.info(f"Cycle {self.cycles_completed} completed successfully")
                else:
                    self.logger.warning(f"Cycle {self.cycles_completed} failed")
                
                # Sleep until next cycle (if not the last one)
                if max_cycles is None or self.cycles_completed < max_cycles:
                    sleep_seconds = cycle_interval_minutes * 60
                    self.logger.info(f"Sleeping for {cycle_interval_minutes} minutes...")
                    await asyncio.sleep(sleep_seconds)
        
        except KeyboardInterrupt:
            self.logger.info("Trading session interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in trading session: {e}")
        
        finally:
            await self._cleanup()
    
    async def _cleanup(self) -> None:
        """Cleanup session resources."""
        session_duration = datetime.now() - self.session_start
        
        self.logger.info("Trading session completed")
        self.logger.info(f"Session duration: {session_duration}")
        self.logger.info(f"Cycles completed: {self.cycles_completed}")
        self.logger.info(f"Successful cycles: {self.cycles_successful}")
        
        if self.cycles_completed > 0:
            success_rate = (self.cycles_successful / self.cycles_completed) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
    
    def get_session_stats(self) -> Dict:
        """
        Get session statistics.
        
        Returns:
            Session statistics dictionary
        """
        session_duration = datetime.now() - self.session_start
        
        return {
            'session_start': self.session_start.isoformat(),
            'session_duration_seconds': session_duration.total_seconds(),
            'cycles_completed': self.cycles_completed,
            'cycles_successful': self.cycles_successful,
            'success_rate': (self.cycles_successful / self.cycles_completed * 100) if self.cycles_completed > 0 else 0,
            'controller_status': self.controller.get_status()
        }
