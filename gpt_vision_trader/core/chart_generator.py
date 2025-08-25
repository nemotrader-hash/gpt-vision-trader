#!/usr/bin/env python3
"""
Chart Generation Core Module
============================

This module handles the generation of candlestick charts with technical indicators
for GPT vision analysis. Refactored from create_dataset.py with improved structure.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from ..utils.indicators import BaseIndicator


class ChartGenerator:
    """
    Generates candlestick charts with technical indicators for GPT analysis.
    Maintains the exact same format as the backtesting system.
    """
    
    # Chart styling constants
    FIGURE_SIZE = (18, 12)
    DPI = 300
    TITLE_FONTSIZE = 15
    TITLE_PAD = 20
    TICK_FONTSIZE = 13
    DATE_FONTSIZE = 13
    PRICE_FONTSIZE = 16
    LINE_WIDTH = 3.0
    
    # Colors
    BULLISH_COLOR = "green"
    BEARISH_COLOR = "red"
    HIDDEN_COLOR = "grey"
    HIDDEN_ALPHA = 1
    
    # Date formatting
    DISPLAYED_DATES_NUMBER = 5
    PLOT_PADDING = 0.02
    
    def __init__(self, 
                 output_dir: str,
                 technical_indicators: Optional[Dict[str, BaseIndicator]] = None):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory to save chart images
            technical_indicators: Dictionary of technical indicators
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.technical_indicators = technical_indicators or {}
        
        logging.info(f"ChartGenerator initialized with output dir: {output_dir}")
    
    def generate_chart(self,
                      ohlcv_data: pd.DataFrame,
                      title: str,
                      hide_after: Optional[pd.Timestamp] = None,
                      filename: Optional[str] = None,
                      full_data_for_indicators: Optional[pd.DataFrame] = None) -> str:
        """
        Generate a candlestick chart with technical indicators.
        
        Args:
            ohlcv_data: OHLCV DataFrame for charting (visible portion)
            title: Chart title
            hide_after: Timestamp after which to hide data (for backtesting format)
            filename: Custom filename (auto-generated if None)
            full_data_for_indicators: Full dataset including buffer for indicator calculation
            
        Returns:
            Path to generated chart image
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"
        
        chart_path = self.output_dir / filename
        
        # Prepare data with indicators
        plot_data = self._prepare_plot_data(ohlcv_data, full_data_for_indicators)
        
        # Create the chart
        self._create_chart(
            plot_data=plot_data,
            title=title,
            output_path=str(chart_path),
            hide_after=hide_after,
            chart_data_for_overlay=ohlcv_data  # Pass original chart data for overlay calculation
        )
        
        logging.info(f"Generated chart: {chart_path}")
        return str(chart_path)
    
    def generate_live_chart(self,
                           visible_data: pd.DataFrame,
                           hidden_placeholder: pd.DataFrame,
                           title: str,
                           pair: str,
                           timeframe: str,
                           full_data_for_indicators: Optional[pd.DataFrame] = None) -> str:
        """
        Generate a live trading chart with hidden future space.
        
        Args:
            visible_data: Historical OHLCV data (visible portion)
            hidden_placeholder: Empty future data placeholder
            title: Chart title
            pair: Trading pair
            timeframe: Timeframe
            full_data_for_indicators: Full dataset including buffer for indicator calculation
            
        Returns:
            Path to generated chart image
        """
        # Combine visible and hidden data - Fix pandas FutureWarning
        if hidden_placeholder.empty:
            combined_data = visible_data.copy()
        else:
            # Handle empty/NaN columns to avoid FutureWarning
            combined_data = pd.concat([visible_data, hidden_placeholder], ignore_index=False, sort=False)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_pair = pair.replace('/', '_')
        filename = f"live_chart_{safe_pair}_{timeframe}_{timestamp}.png"
        
        # Generate chart with hidden overlay at end of visible data
        hide_after = visible_data.index[-1] if not visible_data.empty else None
        
        return self.generate_chart(
            ohlcv_data=combined_data,
            title=title,
            hide_after=hide_after,
            filename=filename,
            full_data_for_indicators=full_data_for_indicators
        )
    
    def _prepare_plot_data(self, ohlcv_df: pd.DataFrame, full_data_for_indicators: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare OHLCV data for plotting by calculating technical indicators.
        
        Args:
            ohlcv_df: OHLCV DataFrame for plotting (visible portion)
            full_data_for_indicators: Full dataset including buffer for indicator calculation
            
        Returns:
            DataFrame with technical indicators calculated (same length as ohlcv_df)
        """
        # Use full data for indicator calculation if provided, otherwise use plot data
        data_for_calculation = full_data_for_indicators if full_data_for_indicators is not None else ohlcv_df
        
        # Ensure proper column names (capitalize for indicators)
        calc_data = data_for_calculation.copy()
        calc_data.columns = [col.capitalize() for col in calc_data.columns]
        
        # Calculate all technical indicators on full dataset
        for indicator in self.technical_indicators.values():
            calc_data = indicator.calculate(calc_data)
        
        # If we used full data for calculation, we need to preserve the structure of ohlcv_df
        if full_data_for_indicators is not None:
            # Start with the original ohlcv_df structure (preserves NaN values in hidden portion)
            plot_data = ohlcv_df.copy()
            
            # Ensure proper column names (capitalize for consistency)
            plot_data.columns = [col.capitalize() for col in plot_data.columns]
            
            # For each timestamp in plot_data, get the indicator values from calc_data
            # This preserves NaN values in the original data while adding indicator columns
            for col in calc_data.columns:
                if col not in plot_data.columns:  # Only add new indicator columns
                    plot_data[col] = np.nan  # Initialize with NaN
                    
                    # Fill indicator values only where we have calculated data
                    for timestamp in plot_data.index:
                        if timestamp in calc_data.index:
                            plot_data.loc[timestamp, col] = calc_data.loc[timestamp, col]
        else:
            plot_data = calc_data
        
        return plot_data
    
    def _create_chart(self,
                     plot_data: pd.DataFrame,
                     title: str,
                     output_path: str,
                     hide_after: Optional[pd.Timestamp] = None,
                     chart_data_for_overlay: Optional[pd.DataFrame] = None) -> None:
        """
        Create the actual chart using mplfinance.
        
        Args:
            plot_data: OHLCV data with indicators
            title: Chart title
            output_path: Path to save chart
            hide_after: Timestamp after which to hide data
            chart_data_for_overlay: Original chart data (visible + hidden, no buffer) for overlay calculation
        """
        # Get indicators that need separate panels
        bottom_indicators = [
            ind for ind in self.technical_indicators.values() 
            if ind.needs_separate_panel
        ]
        n_bottom_panels = len(bottom_indicators)
        
        # Create figure with appropriate layout
        if n_bottom_panels > 0:
            fig = plt.figure(figsize=self.FIGURE_SIZE, constrained_layout=True)
            height_ratios = [3] + [1] * n_bottom_panels
            gs = fig.add_gridspec(
                1 + n_bottom_panels, 1, 
                height_ratios=height_ratios, 
                hspace=0.05
            )
            
            main_ax = fig.add_subplot(gs[0])
            bottom_axes = [
                fig.add_subplot(gs[i], sharex=main_ax) 
                for i in range(1, 1 + n_bottom_panels)
            ]
            
            # Hide x-axis labels except on bottom panel
            main_ax.tick_params(axis='x', labelbottom=False)
            for ax in bottom_axes[:-1]:
                ax.tick_params(axis='x', labelbottom=False)
                
            main_ax.set_title(title, fontsize=self.TITLE_FONTSIZE, pad=self.TITLE_PAD)
        else:
            fig, main_ax = plt.subplots(figsize=self.FIGURE_SIZE, constrained_layout=True)
            main_ax.set_title(title, fontsize=self.TITLE_FONTSIZE, pad=self.TITLE_PAD)
            bottom_axes = []
        
        # Plot main candlestick chart
        mpf.plot(
            plot_data,
            type="candle",
            style=self._create_plot_style(),
            volume=False,
            datetime_format="%Y-%m-%d",
            xrotation=45,
            ax=main_ax,
            show_nontrading=False,
        )
        
        # Plot indicators
        bottom_panel_idx = 0
        for indicator in self.technical_indicators.values():
            if indicator.needs_separate_panel:
                ax = bottom_axes[bottom_panel_idx]
                bottom_panel_idx += 1
            else:
                ax = main_ax
            indicator.plot(ax, plot_data)
        
        # Style main axis
        main_ax.set_ylabel("Price", fontsize=self.PRICE_FONTSIZE)
        main_ax.tick_params(axis="y", labelsize=self.TICK_FONTSIZE)
        
        # Add hidden overlay if specified
        if hide_after is not None:
            self._add_hidden_overlay(plot_data, hide_after, main_ax, chart_data_for_overlay)
            for ax in bottom_axes:
                self._add_hidden_overlay(plot_data, hide_after, ax, chart_data_for_overlay)
        
        # Format date axis
        if bottom_axes:
            self._format_date_axis(plot_data, bottom_axes[-1])
        else:
            self._format_date_axis(plot_data, main_ax)
        
        # Save chart
        plt.savefig(output_path, dpi=self.DPI, bbox_inches="tight")
        plt.close()
    
    def _create_plot_style(self) -> object:
        """Create mplfinance style configuration."""
        market_colors = mpf.make_marketcolors(
            up=self.BULLISH_COLOR,
            down=self.BEARISH_COLOR,
            wick={"up": self.BULLISH_COLOR, "down": self.BEARISH_COLOR},
            edge={"up": self.BULLISH_COLOR, "down": self.BEARISH_COLOR},
        )
        
        return mpf.make_mpf_style(
            marketcolors=market_colors,
            gridstyle="--",
            y_on_right=False,
            rc={
                "font.size": self.TICK_FONTSIZE,
                "axes.labelsize": self.PRICE_FONTSIZE,
                "lines.linewidth": self.LINE_WIDTH,
            },
        )
    
    def _add_hidden_overlay(self, 
                           df: pd.DataFrame, 
                           hide_after: pd.Timestamp, 
                           ax: plt.Axes,
                           chart_data_only: Optional[pd.DataFrame] = None) -> None:
        """
        Add grey overlay after specified timestamp.
        
        Args:
            df: Full plot data (may include buffer days)
            hide_after: Timestamp after which to hide data
            ax: Matplotlib axes
            chart_data_only: Only the data being charted (visible + hidden, no buffer)
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # Use chart_data_only if provided (this excludes buffer days)
        # Otherwise fall back to df for backward compatibility
        calculation_df = chart_data_only if chart_data_only is not None else df
        
        # Find visible portion
        visible_mask = calculation_df.index <= hide_after
        if not visible_mask.any():
            return
        
        visible_df = calculation_df[visible_mask]
        hide_ratio = len(visible_df) / len(calculation_df)
        
        # Add grey rectangle for hidden portion
        start_x = xmin + (xmax - xmin) * hide_ratio
        rect = plt.Rectangle(
            (start_x, ymin),
            xmax - start_x,
            ymax - ymin,
            facecolor=self.HIDDEN_COLOR,
            alpha=self.HIDDEN_ALPHA,
            zorder=100,
        )
        ax.add_patch(rect)
    
    def _format_date_axis(self, ohlcv_df: pd.DataFrame, ax: plt.Axes) -> None:
        """Format x-axis with proper date labels."""
        indices = list(range(len(ohlcv_df)))
        
        if self.DISPLAYED_DATES_NUMBER <= 2:
            tick_positions = [indices[0], indices[-1]]
        else:
            step = (len(indices) - 1) / (self.DISPLAYED_DATES_NUMBER - 1)
            tick_positions = [int(i * step) for i in range(self.DISPLAYED_DATES_NUMBER)]
            tick_positions[-1] = indices[-1]
        
        ax.set_xticks(tick_positions)
        
        tick_labels = [
            ohlcv_df.index[pos].strftime("%Y-%m-%d") 
            for pos in tick_positions
        ]
        
        ax.set_xticklabels(
            tick_labels, 
            rotation=45, 
            ha="right", 
            fontsize=self.DATE_FONTSIZE
        )
        ax.tick_params(axis='x', labelsize=self.DATE_FONTSIZE)
        
        # Add padding
        padding = len(indices) * self.PLOT_PADDING
        ax.set_xlim(indices[0] - padding, indices[-1] + padding)
    
    def cleanup_old_charts(self, keep_last_n: int = 10) -> None:
        """
        Clean up old chart files to prevent disk space issues.
        
        Args:
            keep_last_n: Number of recent charts to keep
        """
        chart_files = list(self.output_dir.glob("*.png"))
        
        if len(chart_files) <= keep_last_n:
            return
        
        # Sort by modification time and remove oldest
        chart_files.sort(key=lambda x: x.stat().st_mtime)
        files_to_remove = chart_files[:-keep_last_n]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logging.debug(f"Removed old chart: {file_path}")
            except Exception as e:
                logging.warning(f"Failed to remove old chart {file_path}: {e}")
