import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import matplotlib.pyplot as plt
import mplfinance as mpf
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
    
    def __init__(
        self, 
        period: int = 20,
        color: str = '#1f77b4',
        alpha: float = 0.8,
        linewidth: float = 1.0
    ):
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
    
    def __init__(
        self, 
        period: int = 14,
        color: str = '#8757ff',  # Purple
        alpha: float = 0.8,
        linewidth: float = 1.0,
        overbought: float = 70,
        oversold: float = 30
    ):
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
        ax.set_ylabel('RSI', fontsize=ChartPlotter.PRICE_FONTSIZE)  # Same size as price label
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=ChartPlotter.TICK_FONTSIZE)  # Same size as other labels
        ax.tick_params(axis='both', labelsize=ChartPlotter.TICK_FONTSIZE)  # Same size as other ticks
    
    @property
    def column_names(self) -> List[str]:
        """Get list of column names added by this indicator."""
        return [f'RSI_{self.period}']

    @property
    def needs_separate_panel(self) -> bool:
        return True


class MACDIndicator(BaseIndicator):
    """Moving Average Convergence Divergence indicator."""
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        color: str = '#2ecc71',  # Green for histogram
        signal_color: str = '#e74c3c',  # Red for signal
        macd_color: str = '#3498db',  # Blue for MACD
        alpha: float = 0.8,
        linewidth: float = 1.0
    ):
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
        ax.set_ylabel('MACD', fontsize=ChartPlotter.PRICE_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=ChartPlotter.TICK_FONTSIZE)
        ax.tick_params(axis='both', labelsize=ChartPlotter.TICK_FONTSIZE)
    
    @property
    def column_names(self) -> List[str]:
        """Get list of column names added by this indicator."""
        return ['MACD_line', 'MACD_signal', 'MACD_hist']


class ATRIndicator(BaseIndicator):
    """Average True Range indicator."""
    
    def __init__(
        self, 
        period: int = 14,
        color: str = '#e67e22',  # Orange
        alpha: float = 0.8,
        linewidth: float = 1.0
    ):
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
        ax.set_ylabel('ATR', fontsize=ChartPlotter.PRICE_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=ChartPlotter.TICK_FONTSIZE)
        ax.tick_params(axis='both', labelsize=ChartPlotter.TICK_FONTSIZE)
    
    @property
    def column_names(self) -> List[str]:
        """Get list of column names added by this indicator."""
        return [f'ATR_{self.period}']


class DateManager:
    BUFFER_DAYS: int = 10

    def __init__(
        self,
        start_year_month: str,
        end_year_month: str,
        visible_days: int,
        hidden_days: int,
    ):
        """
        Initialize the DateManager with user parameters.

        Args:
            start_year_month: Start date in YYYY-MM format
            end_year_month: End date in YYYY-MM format
            visible_days: Number of visible days per window
            hidden_days: Number of hidden days per window
        """
        self._start_year_month = start_year_month
        self._end_year_month = end_year_month
        self._visible_days = visible_days
        self._hidden_days = hidden_days

        self._windows: List[Dict] = []
        self._download_start: Optional[str] = None
        self._download_end: Optional[str] = None

        self._generate_all_dates()

    @property
    def windows(self) -> List[Dict]:
        return self._windows

    @property
    def download_range(self) -> Tuple[str, str]:
        return self._download_start, self._download_end

    def _generate_all_dates(self) -> None:
        """Generate all date information including windows and download range."""
        self._windows = self._generate_window_dates()
        self._download_start, self._download_end = self._calculate_download_range()

    def _generate_window_dates(self) -> List[Dict]:
        """
        Generate a list of dictionaries containing start, visible end, and end dates.
        Windows roll by hidden_days, creating overlapping sets that cover all data.
        """
        end_date_dt = pd.to_datetime(self._get_month_end_date(self._end_year_month))
        current_date = pd.to_datetime(f"{self._start_year_month}-01")

        windows = []
        window_id = 1

        while current_date <= end_date_dt:
            visible_end = current_date + pd.Timedelta(days=self._visible_days - 1)
            window_end = visible_end + pd.Timedelta(days=self._hidden_days)

            if window_end <= end_date_dt:
                window = {
                    "id": window_id,
                    "start_date": current_date.strftime("%Y-%m-%d"),
                    "visible_end_date": visible_end.strftime("%Y-%m-%d"),
                    "end_date": window_end.strftime("%Y-%m-%d"),
                }
                windows.append(window)
                window_id += 1

                current_date = current_date + pd.Timedelta(days=self._hidden_days)
            else:
                break

        return windows

    def _calculate_download_range(self) -> Tuple[str, str]:
        """Calculate the date range for downloading OHLCV data with buffer days."""
        if not self._windows:
            return None, None

        first_start = pd.to_datetime(self._windows[0]["start_date"])
        last_end = pd.to_datetime(self._windows[-1]["end_date"])

        download_start = (first_start - pd.Timedelta(days=self.BUFFER_DAYS)).strftime(
            "%Y-%m-%d"
        )
        download_end = (last_end + pd.Timedelta(days=self.BUFFER_DAYS)).strftime(
            "%Y-%m-%d"
        )

        return download_start, download_end

    def _get_month_end_date(self, year_month: str) -> str:
        """Get the last day of a month in YYYY-MM format."""
        month_start = datetime.strptime(f"{year_month}-01", "%Y-%m-%d")
        next_month = month_start.replace(day=28) + pd.Timedelta(days=4)
        month_end = next_month - pd.Timedelta(days=next_month.day)
        return month_end.strftime("%Y-%m-%d")

    def print_summary(self) -> None:
        """Print a summary of all generated dates."""
        print("\n > Generated window dates:")

        for window in self._windows[:2]:
            print(
                f"Window {window['id']}: {window['start_date']} → "
                f"{window['visible_end_date']} (visible) → {window['end_date']} (hidden)"
            )

        if len(self._windows) > 4:
            print("⋮")  # Vertical dots

        for window in self._windows[-2:]:
            print(
                f"Window {window['id']}: {window['start_date']} → "
                f"{window['visible_end_date']} (visible) → {window['end_date']} (hidden)"
            )

        print(f"\nTotal number of windows: {len(self._windows)}")

        print("\n > Download range with buffer:")
        print(f"Download start date: {self._download_start}")
        print(f"Download end date: {self._download_end}")


class ChartPlotter:
    FIGURE_SIZE = (18, 12)
    DPI = 300

    TITLE_FONTSIZE = 15
    TITLE_PAD = 20
    TICK_FONTSIZE = 13
    DATE_FONTSIZE = 13
    PRICE_FONTSIZE = 16

    LINE_WIDTH = 3.0

    BULLISH_COLOR = "green"
    BEARISH_COLOR = "red"
    HIDDING_COLOR = "grey"
    HIDDING_ALPHA = 1

    FORCE_PLOT_DATE_RANGE = True
    DISPLAYED_DATES_NUMBER = 5
    PLOT_PADDING = 0.02

    def __init__(
        self, 
        ohlcv_df: pd.DataFrame, 
        output_dir: str,
        technical_indicators: Optional[Dict[str, BaseIndicator]] = None
    ):
        """
        Initialize with OHLCV data and output directory.

        Args:
            ohlcv_df: DataFrame with OHLCV data
            output_dir: Directory to save chart images
            technical_indicators: Optional dictionary of technical indicators with their names as keys
        """
        self._plot_dir = Path(output_dir)
        self._plot_dir.mkdir(parents=True, exist_ok=True)
        self._technical_indicators = technical_indicators or {}
        self._ohlcv_df = self._calculate_indicators(ohlcv_df)

    def create_plot_pair(self, window: Dict, title: str) -> Tuple[str, str]:
        """
        Create hidden and result plots for a window of data.

        Args:
            window: Dictionary with window date information (start_date, visible_end_date, end_date)
            title: Plot title

        Returns:
            Tuple of (hidden_filename, result_filename)
        """
        start_date = pd.to_datetime(window["start_date"])
        visible_end = pd.to_datetime(window["visible_end_date"])
        end_date = pd.to_datetime(window["end_date"])

        ohlcv_df = self._ohlcv_df[
            (self._ohlcv_df.index >= start_date) & (self._ohlcv_df.index <= end_date)
        ]

        hidden_filename, result_filename = self._generate_filenames(
            window["start_date"], window["end_date"]
        )

        self._create_plot(
            ohlcv_df=ohlcv_df,
            title=title,
            output_path=str(self._plot_dir / hidden_filename),
            hide_after=visible_end,
        )

        self._create_plot(
            ohlcv_df=ohlcv_df,
            title=title,
            output_path=str(self._plot_dir / result_filename),
            hide_after=None,
        )

        return hidden_filename, result_filename

    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators on the full dataset.
        
        Args:
            df: Original OHLCV DataFrame
            
        Returns:
            DataFrame with all indicators calculated
        """
        result_df = ohlcv_df.copy()
        result_df.columns = [col.capitalize() for col in result_df.columns]
        
        for indicator in self._technical_indicators.values():
            result_df = indicator.calculate(result_df)
            
        return result_df
    
    def _generate_filenames(self, start_date: str, end_date: str) -> Tuple[str, str]:
        """Generate filenames for hidden and result plots."""
        hidden_filename = f"chart_{start_date}_{end_date}_hidden.png"
        result_filename = f"chart_{start_date}_{end_date}_result.png"
        return hidden_filename, result_filename

    def _create_plot(
        self,
        ohlcv_df: pd.DataFrame,
        title: str,
        output_path: str,
        hide_after: Optional[pd.Timestamp] = None,
    ) -> None:
        """Create a single plot from OHLCV data."""
        plot_ohlcv_df = ohlcv_df.copy()

        # Get all indicators that need separate panels
        bottom_indicators = [ind for ind in self._technical_indicators.values() if ind.needs_separate_panel]
        n_bottom_panels = len(bottom_indicators)
        
        if n_bottom_panels > 0:
            fig = plt.figure(figsize=self.FIGURE_SIZE, constrained_layout=True)
            
            height_ratios = [3] + [1] * n_bottom_panels
            gs = fig.add_gridspec(1 + n_bottom_panels, 1, height_ratios=height_ratios, hspace=0.05)
            
            main_ax = fig.add_subplot(gs[0])
            bottom_axes = [fig.add_subplot(gs[i], sharex=main_ax) for i in range(1, 1 + n_bottom_panels)]
            
            main_ax.tick_params(axis='x', labelbottom=False)
            for ax in bottom_axes[:-1]:
                ax.tick_params(axis='x', labelbottom=False)
            
            main_ax.set_title(title, fontsize=self.TITLE_FONTSIZE, pad=self.TITLE_PAD)
        else:
            fig, main_ax = plt.subplots(figsize=self.FIGURE_SIZE, constrained_layout=True)
            main_ax.set_title(title, fontsize=self.TITLE_FONTSIZE, pad=self.TITLE_PAD)
            bottom_axes = []

        # Plot main chart
        mpf.plot(
            plot_ohlcv_df,
            type="candle",
            style=self._create_plot_style(),
            volume=False,
            datetime_format="%Y-%m-%d",
            xrotation=45,
            ax=main_ax,
            show_nontrading=False,
        )
        
        bottom_panel_idx = 0
        for indicator in self._technical_indicators.values():
            if indicator.needs_separate_panel:
                ax = bottom_axes[bottom_panel_idx]
                bottom_panel_idx += 1
            else:
                ax = main_ax
            indicator.plot(ax, plot_ohlcv_df)

        main_ax.set_ylabel("Price", fontsize=self.PRICE_FONTSIZE)
        main_ax.tick_params(axis="y", labelsize=self.TICK_FONTSIZE)

        if hide_after is not None:
            self._add_hidden_overlay(plot_ohlcv_df, hide_after, main_ax)
            for ax in bottom_axes:
                self._add_hidden_overlay(plot_ohlcv_df, hide_after, ax)

        if self.FORCE_PLOT_DATE_RANGE:
            if bottom_axes:
                # Format dates only on the last bottom panel
                self._format_date_axis(plot_ohlcv_df, bottom_axes[-1])
            else:
                self._format_date_axis(plot_ohlcv_df, main_ax)

        plt.savefig(output_path, dpi=self.DPI, bbox_inches="tight")
        plt.close()

    def _create_plot_style(self) -> object:
        """Create the plot style configuration."""
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

    def _add_hidden_overlay(self, df: pd.DataFrame, hide_after: pd.Timestamp, ax: plt.Axes) -> None:
        """Add grey overlay after specified timestamp."""
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Find the closest timestamp that's less than or equal to hide_after
        # This handles cases where hide_after doesn't exactly match a candle timestamp
        visible_mask = df.index <= hide_after
        if not visible_mask.any():
            # If no visible data, don't add overlay
            return
            
        visible_df = df[visible_mask]
        hide_ratio = len(visible_df) / len(df)

        start_x = xmin + (xmax - xmin) * hide_ratio
        rect = plt.Rectangle(
            (start_x, ymin),
            xmax - start_x,
            ymax - ymin,
            facecolor=self.HIDDING_COLOR,
            alpha=self.HIDDING_ALPHA,
            zorder=100,
        )
        ax.add_patch(rect)

    def _format_date_axis(self, ohlcv_df: pd.DataFrame, ax: plt.Axes) -> None:
        """Format x-axis to ensure window dates are shown correctly."""
        indices = list(range(len(ohlcv_df)))

        if self.DISPLAYED_DATES_NUMBER <= 2:
            tick_positions = [indices[0], indices[-1]]
        else:
            step = (len(indices) - 1) / (self.DISPLAYED_DATES_NUMBER - 1)
            tick_positions = [int(i * step) for i in range(self.DISPLAYED_DATES_NUMBER)]
            tick_positions[-1] = indices[-1]

        ax.set_xticks(tick_positions)

        tick_labels = [
            ohlcv_df.index[pos].strftime("%Y-%m-%d") for pos in tick_positions
        ]

        ax.set_xticklabels(
            tick_labels, 
            rotation=45, 
            ha="right", 
            fontsize=self.DATE_FONTSIZE
        )
        ax.tick_params(axis='x', labelsize=self.DATE_FONTSIZE)

        padding = len(indices) * self.PLOT_PADDING
        ax.set_xlim(indices[0] - padding, indices[-1] + padding)


class OHLCVManager:
    """Manages OHLCV data operations including fetching and window data extraction."""
    
    def __init__(self, symbol: str, timeframe: str, exchange_name: str = "binance"):
        """
        Initialize the OHLCV manager.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "4h")
            exchange_name: Exchange ID (default: 'binance')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_name = exchange_name
        self._data: Optional[pd.DataFrame] = None
        
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data
        """
        exchange = getattr(ccxt, self.exchange_name)()
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_ohlcv = []
        current_timestamp = start_timestamp
        df = None
        while current_timestamp < end_timestamp:
            try:
                ohlcv = exchange.fetch_ohlcv(
                        self.symbol, self.timeframe, since=current_timestamp, limit=1000
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                current_timestamp = ohlcv[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                print(f"Error fetching data: {e}")
                break

            df = pd.DataFrame(
                    all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            self._data = df
        if df is not None:
            return df
        else:
            raise ValueError("No data available. Call fetch_data first.")

    def get_window_data(self, window: Dict) -> Dict:
        """
        Extract and format OHLCV data for both visible and hidden portions of a window.
        
        Args:
            window: Dictionary containing window information (start_date, visible_end_date, end_date)
            
        Returns:
            Dictionary containing formatted OHLCV data for visible and hidden portions
        """
        if self._data is None:
            raise ValueError("No data available. Call fetch_data first.")
            
        start_date = pd.to_datetime(window["start_date"])
        visible_end = pd.to_datetime(window["visible_end_date"])
        end_date = pd.to_datetime(window["end_date"])
        
        window_df = self._data[
            (self._data.index >= start_date) & (self._data.index <= end_date)
        ]
        visible_df = window_df[window_df.index <= visible_end]
        hidden_df = window_df[window_df.index > visible_end]
        
        return {
            "visible_candles": self._format_candles(visible_df),
            "hidden_candles": self._format_candles(hidden_df)
        }
    
    def calculate_performance(self, window: Dict) -> float:
        """
        Calculate price performance between the last visible candle and the last hidden candle.
        
        Args:
            window: Dictionary with window information
            
        Returns:
            Performance as decimal
        """
        if self._data is None:
            raise ValueError("No data available. Call fetch_data first.")
            
        visible_end = pd.to_datetime(window["visible_end_date"])
        window_end = pd.to_datetime(window["end_date"])

        visible_end_close = self._data.loc[visible_end, "close"]
        window_end_close = self._data.loc[window_end, "close"]

        performance = (window_end_close / visible_end_close) - 1
        return round(performance, 4)

    def calculate_trend(self, window: Dict) -> Dict[str, float | str]:
        """
        Calculate trend information based on ATR at the last visible candle.
        
        Args:
            window: Dictionary with window information
            
        Returns:
            Dictionary containing:
                - atr_factor: How many ATR units away the last hidden candle is
                - trend: 'bullish', 'bearish', or 'neutral'
                - atr: The ATR value at the last visible candle
        """
        if self._data is None:
            raise ValueError("No data available. Call fetch_data first.")
            
        visible_end = pd.to_datetime(window["visible_end_date"])
        window_end = pd.to_datetime(window["end_date"])
        
        period = 14
        high_low = self._data['high'] - self._data['low']
        high_close = abs(self._data['high'] - self._data['close'].shift())
        low_close = abs(self._data['low'] - self._data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        visible_atr = atr.loc[visible_end]
        
        visible_close = self._data.loc[visible_end, "close"]
        hidden_close = self._data.loc[window_end, "close"]
        price_change = hidden_close - visible_close
        atr_factor = price_change / visible_atr
        
        trend = 'neutral'
        if atr_factor > 1:
            trend = 'bullish'
        elif atr_factor < -1:
            trend = 'bearish'
            
        return {
            'atr_factor': round(atr_factor, 4),
            'trend': trend,
            'atr': round(visible_atr, 4)
        }

    @staticmethod
    def _format_candles(df: pd.DataFrame) -> list[dict]:
        """
        Format OHLCV data into a list of candle dictionaries.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of formatted candle dictionaries
        """
        return [{
            "time": int(t.timestamp() * 1000),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        } for t, row in df.iterrows()]
        

class TechnicalIndicator:
    """Class for calculating and plotting technical indicators."""
    
    def __init__(self, 
                 sma_periods: List[int] = [20, 50, 200],
                 sma_color: str = '#1f77b4',
                 sma_alpha: float = 0.8,
                 sma_linewidth: float = 1.0):
        """
        Initialize technical indicator calculator.
        
        Args:
            sma_periods: List of periods for SMA calculation
            sma_color: Color for SMA lines
            sma_alpha: Transparency for SMA lines
            sma_linewidth: Width of SMA lines
        """
        self.sma_periods = sma_periods
        self.sma_color = sma_color
        self.sma_alpha = sma_alpha
        self.sma_linewidth = sma_linewidth
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with added indicator columns
        """
        result_df = df.copy()
        
        # Calculate SMAs
        for period in self.sma_periods:
            col_name = f'sma_{period}'
            result_df[col_name] = result_df['Close'].rolling(window=period).mean()
            
        return result_df
    
    def plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """
        Plot technical indicators on the given axes.
        
        Args:
            ax: Matplotlib axes to plot on
            df: DataFrame containing the indicator data
        """
        # Plot SMAs
        for period in self.sma_periods:
            col_name = f'sma_{period}'
            if col_name in df.columns:
                ax.plot(
                    range(len(df)),
                    df[col_name],
                    label=f'SMA {period}',
                    color=self.sma_color,
                    alpha=self.sma_alpha,
                    linewidth=self.sma_linewidth
                )
        
        # Add legend if we plotted any indicators
        if self.sma_periods:
            ax.legend(loc='upper left')


def create_dataset(
    symbol: str,
    timeframe: str,
    start_year_month: str,
    end_year_month: str,
    visible_days: int = 3,
    hidden_days: int = 1,
    output_dir: str = "dataset",
    technical_indicators: Optional[Dict[str, BaseIndicator]] = None,
) -> None:
    """Create a dataset of charts with hidden and result plots using DateManager and ChartPlotter.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        timeframe: Candle timeframe (e.g., "4h")
        start_year_month: Start date in YYYY-MM format
        end_year_month: End date in YYYY-MM format
        visible_days: Number of visible days per chart
        hidden_days: Number of hidden days per chart
        output_dir: Directory to save the dataset
        technical_indicators: Optional dictionary of technical indicators to add to charts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    date_manager = DateManager(start_year_month, end_year_month, visible_days, hidden_days)
    ohlcv_manager = OHLCVManager(symbol, timeframe)
    
    ohlcv_df = ohlcv_manager.fetch_data(*date_manager.download_range)
    plotter = ChartPlotter(ohlcv_df, str(plots_dir), technical_indicators)
    
    metadata = {}
    metadata2 = {}  # Second metadata without OHLCV data
    total_plots = len(date_manager.windows)
    
    title = f"{symbol} {timeframe}"
    
    for i, window in enumerate(date_manager.windows, 1):
        hidden_file, result_file = plotter.create_plot_pair(window, title)
        perf = ohlcv_manager.calculate_performance(window)
        trend_info = ohlcv_manager.calculate_trend(window)
        ohlcv_data = ohlcv_manager.get_window_data(window)
        
        print(f"Creating plot {i}/{total_plots}")
        
        # Full metadata with OHLCV data
        metadata[hidden_file] = {
            "result_file": result_file,
            "start_date": window["start_date"],
            "visible_end_date": window["visible_end_date"],
            "end_date": window["end_date"],
            "performance": perf,
            "trend": trend_info["trend"],
            "atr_factor": trend_info["atr_factor"],
            "atr": trend_info["atr"],
            **ohlcv_data
        }
        
        # Metadata without OHLCV data
        metadata2[hidden_file] = {
            "result_file": result_file,
            "start_date": window["start_date"],
            "visible_end_date": window["visible_end_date"],
            "end_date": window["end_date"],
            "performance": perf,
            "trend": trend_info["trend"],
            "atr_factor": trend_info["atr_factor"],
            "atr": trend_info["atr"]
        }
    
    # Save both metadata files
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    with open(output_path / "metadata2.json", "w") as f:
        json.dump(metadata2, f, indent=4)


if __name__ == "__main__":
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "4h"
    START_YEAR_MONTH = "2023-01"
    END_YEAR_MONTH = "2024-01"
    VISIBLE_DAYS = 3
    HIDDEN_DAYS = 1
    OUTPUT_DIR = "btc_4h_dataset"

    indicators: Dict[str, BaseIndicator] = {
        "SMA20": SMAIndicator(
            period=20,
            color='#ff7f0e',  
            alpha=0.8,
            linewidth=2.0
        ),
        "RSI14": RSIIndicator(
            period=14,
            color='#8757ff', 
            alpha=0.8,
            linewidth=1.0,
            overbought=70,
            oversold=30
        ),
        "MACD": MACDIndicator(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            color='#2ecc71', 
            signal_color='#e74c3c',  
            macd_color='#3498db',
            alpha=0.8,
            linewidth=1.0
        ),
        "ATR14": ATRIndicator(
            period=14,
            color='#e67e22', 
            alpha=0.8,
            linewidth=1.0
        )
    }

    create_dataset(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_year_month=START_YEAR_MONTH,
        end_year_month=END_YEAR_MONTH,
        visible_days=VISIBLE_DAYS,
        hidden_days=HIDDEN_DAYS,
        output_dir=OUTPUT_DIR,
        technical_indicators=indicators
    )


