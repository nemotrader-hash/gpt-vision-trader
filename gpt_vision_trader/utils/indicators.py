#!/usr/bin/env python3
"""
Technical Indicators Module
===========================

This module provides technical indicators for chart generation.
Refactored from create_dataset.py with improved structure and organization.
"""

from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


class BaseIndicator(ABC):
    """Abstract base class for technical indicators."""
    
    def __init__(self, color: str, alpha: float = 0.8, linewidth: float = 1.0):
        """
        Initialize base indicator properties.
        
        Args:
            color: Line color in hex format
            alpha: Line transparency (0-1)
            linewidth: Width of the line
        """
        self.color = color
        self.alpha = alpha
        self.linewidth = linewidth
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values.
        
        Args:
            df: OHLCV DataFrame with capitalized column names
            
        Returns:
            DataFrame with added indicator columns
        """
        pass
    
    @abstractmethod
    def plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """
        Plot the indicator on the given axes.
        
        Args:
            ax: Matplotlib axes to plot on
            df: DataFrame containing the indicator data
        """
        pass
    
    @property
    @abstractmethod
    def column_names(self) -> List[str]:
        """Get list of column names this indicator adds to the DataFrame."""
        pass
        
    @property
    def needs_separate_panel(self) -> bool:
        """Whether this indicator should be plotted in a separate panel below the main chart."""
        return False


class SMAIndicator(BaseIndicator):
    """Simple Moving Average indicator."""
    
    def __init__(self, 
                 period: int = 20,
                 color: str = '#1f77b4',
                 alpha: float = 0.8,
                 linewidth: float = 1.0):
        """
        Initialize SMA indicator.
        
        Args:
            period: Period for calculating SMA
            color: Line color in hex format
            alpha: Line transparency (0-1)
            linewidth: Width of the line
        """
        super().__init__(color, alpha, linewidth)
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA value."""
        result_df = df.copy()
        col_name = f'SMA_{self.period}'
        result_df[col_name] = result_df['Close'].rolling(window=self.period).mean()
        return result_df
    
    def plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot SMA line on the given axes."""
        col_name = f'SMA_{self.period}'
        if col_name in df.columns:
            ax.plot(
                range(len(df)),
                df[col_name],
                label=f'SMA {self.period}',
                color=self.color,
                alpha=self.alpha,
                linewidth=self.linewidth
            )
            ax.legend(loc='upper left')
    
    @property
    def column_names(self) -> List[str]:
        """Get list of column names added by this indicator."""
        return [f'SMA_{self.period}']


class RSIIndicator(BaseIndicator):
    """Relative Strength Index indicator."""
    
    def __init__(self, 
                 period: int = 14,
                 color: str = '#8757ff',
                 alpha: float = 0.8,
                 linewidth: float = 1.0,
                 overbought: float = 70,
                 oversold: float = 30):
        """
        Initialize RSI indicator.
        
        Args:
            period: Period for RSI calculation
            color: Line color in hex format
            alpha: Line transparency (0-1)
            linewidth: Width of the line
            overbought: Overbought level (default 70)
            oversold: Oversold level (default 30)
        """
        super().__init__(color, alpha, linewidth)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI values."""
        result_df = df.copy()
        
        delta = result_df['Close'].diff()
        
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result_df[f'RSI_{self.period}'] = rsi
        return result_df
    
    def plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot RSI in a subplot panel."""
        col_name = f'RSI_{self.period}'
        if col_name not in df.columns:
            return
            
        # Plot RSI line
        ax.plot(
            range(len(df)),
            df[col_name],
            label=f'RSI ({self.period})',
            color=self.color,
            alpha=self.alpha,
            linewidth=self.linewidth
        )
        
        # Add overbought/oversold lines
        ax.axhline(y=self.overbought, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=self.oversold, color='g', linestyle='--', alpha=0.5)
        
        # Set RSI panel properties
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
    
    @property
    def column_names(self) -> List[str]:
        """Get list of column names added by this indicator."""
        return [f'RSI_{self.period}']

    @property
    def needs_separate_panel(self) -> bool:
        return True


class MACDIndicator(BaseIndicator):
    """Moving Average Convergence Divergence indicator."""
    
    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 color: str = '#2ecc71',
                 signal_color: str = '#e74c3c',
                 macd_color: str = '#3498db',
                 alpha: float = 0.8,
                 linewidth: float = 1.0):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            color: Color for the histogram
            signal_color: Color for the signal line
            macd_color: Color for the MACD line
            alpha: Line transparency
            linewidth: Width of the lines
        """
        super().__init__(color, alpha, linewidth)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.signal_color = signal_color
        self.macd_color = macd_color
    
    @property
    def needs_separate_panel(self) -> bool:
        return True
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD values."""
        result_df = df.copy()
        
        fast_ema = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        
        result_df['MACD_line'] = fast_ema - slow_ema
        result_df['MACD_signal'] = result_df['MACD_line'].ewm(span=self.signal_period, adjust=False).mean()
        result_df['MACD_hist'] = result_df['MACD_line'] - result_df['MACD_signal']
        
        return result_df
    
    def plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot MACD in a subplot panel."""
        if 'MACD_line' not in df.columns:
            return
            
        # Plot histogram
        ax.bar(
            range(len(df)),
            df['MACD_hist'],
            color=self.color,
            alpha=self.alpha,
            label='Histogram'
        )
        
        # Plot MACD and Signal lines
        ax.plot(
            range(len(df)),
            df['MACD_line'],
            color=self.macd_color,
            alpha=self.alpha,
            linewidth=self.linewidth,
            label='MACD'
        )
        ax.plot(
            range(len(df)),
            df['MACD_signal'],
            color=self.signal_color,
            alpha=self.alpha,
            linewidth=self.linewidth,
            label='Signal'
        )
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set panel properties
        ax.set_ylabel('MACD', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
    
    @property
    def column_names(self) -> List[str]:
        """Get list of column names added by this indicator."""
        return ['MACD_line', 'MACD_signal', 'MACD_hist']


class ATRIndicator(BaseIndicator):
    """Average True Range indicator."""
    
    def __init__(self, 
                 period: int = 14,
                 color: str = '#e67e22',
                 alpha: float = 0.8,
                 linewidth: float = 1.0):
        """
        Initialize ATR indicator.
        
        Args:
            period: Period for ATR calculation
            color: Line color in hex format
            alpha: Line transparency (0-1)
            linewidth: Width of the line
        """
        super().__init__(color, alpha, linewidth)
        self.period = period
    
    @property
    def needs_separate_panel(self) -> bool:
        return True
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR values."""
        result_df = df.copy()
        
        high_low = result_df['High'] - result_df['Low']
        high_close = abs(result_df['High'] - result_df['Close'].shift())
        low_close = abs(result_df['Low'] - result_df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result_df[f'ATR_{self.period}'] = true_range.rolling(window=self.period).mean()
        
        return result_df
    
    def plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot ATR in a subplot panel."""
        col_name = f'ATR_{self.period}'
        if col_name not in df.columns:
            return
            
        # Plot ATR line
        ax.plot(
            range(len(df)),
            df[col_name],
            label=f'ATR ({self.period})',
            color=self.color,
            alpha=self.alpha,
            linewidth=self.linewidth
        )
        
        # Set panel properties
        ax.set_ylabel('ATR', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
    
    @property
    def column_names(self) -> List[str]:
        """Get list of column names added by this indicator."""
        return [f'ATR_{self.period}']
