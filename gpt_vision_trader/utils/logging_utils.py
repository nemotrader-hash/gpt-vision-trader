#!/usr/bin/env python3
"""
Logging Utilities Module
========================

This module provides centralized logging configuration and utilities
for the GPT Vision Trading system.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 log_dir: str = "gpt_vision_trader/data/logs") -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Specific log file path (optional)
        log_dir: Directory for log files
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(log_dir, f"gpt_vision_trader_{timestamp}.log")
    
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized - Console: {logging.getLevelName(level)}, File: {file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class TradingLogHandler:
    """
    Specialized log handler for trading activities.
    Logs trading decisions, analysis results, and system events.
    """
    
    def __init__(self, log_dir: str = "gpt_vision_trader/data/trading_logs"):
        """
        Initialize trading log handler.
        
        Args:
            log_dir: Directory for trading logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"trading_{today}.jsonl"
        
    def log_analysis(self, 
                    chart_path: str,
                    prediction: str,
                    analysis: str,
                    pair: str,
                    timeframe: str) -> None:
        """
        Log GPT analysis results.
        
        Args:
            chart_path: Path to analyzed chart
            prediction: GPT prediction
            analysis: GPT analysis text
            pair: Trading pair
            timeframe: Timeframe
        """
        import json
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "analysis",
            "chart_path": chart_path,
            "prediction": prediction,
            "analysis": analysis,
            "pair": pair,
            "timeframe": timeframe
        }
        
        self._write_log_entry(log_entry)
    
    def log_trade_signal(self,
                        signals: dict,
                        pair: str,
                        prediction: str) -> None:
        """
        Log trading signals.
        
        Args:
            signals: Trading signals dictionary
            pair: Trading pair
            prediction: GPT prediction that generated signals
        """
        import json
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "signals",
            "signals": signals,
            "pair": pair,
            "prediction": prediction
        }
        
        self._write_log_entry(log_entry)
    
    def log_trade_execution(self,
                          action: str,
                          pair: str,
                          success: bool,
                          details: dict = None) -> None:
        """
        Log trade execution results.
        
        Args:
            action: Trade action (enter_long, exit_long, etc.)
            pair: Trading pair
            success: Whether execution was successful
            details: Additional execution details
        """
        import json
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "execution",
            "action": action,
            "pair": pair,
            "success": success,
            "details": details or {}
        }
        
        self._write_log_entry(log_entry)
    
    def _write_log_entry(self, log_entry: dict) -> None:
        """Write log entry to file."""
        import json
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logging.error(f"Failed to write trading log entry: {e}")


def setup_trading_logging(log_dir: str = "gpt_vision_trader/data") -> TradingLogHandler:
    """
    Setup trading-specific logging.
    
    Args:
        log_dir: Base directory for logs
        
    Returns:
        Trading log handler instance
    """
    trading_log_dir = os.path.join(log_dir, "trading_logs")
    return TradingLogHandler(trading_log_dir)
