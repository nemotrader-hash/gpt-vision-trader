import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_date: str
    entry_price: float
    side: str 
    size: float
    entry_fee: float
    entry_cashflow: float 
    exit_fee: Optional[float] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_cashflow: Optional[float] = None  
    pnl: Optional[float] = None

@dataclass
class Position:
    side: str  
    entry_price: float
    entry_date: str
    size: float 
    entry_fee: float  
    entry_cashflow: float 
    
    @classmethod
    def open(cls, side: str, entry_price: float, entry_date: str, 
             cost: float, fee_rate: float) -> 'Position':
        """
        Open a new position with cost and calculate actual size after fees.
        
        Args:
            side: 'long' or 'short'
            entry_price: Price per unit
            entry_date: Entry timestamp
            cost: Amount in quote currency to spend (e.g., 1000 USDT)
            fee_rate: Fee rate as decimal
        """
        # For the same cost, we get less size due to fee
        entry_fee = cost * fee_rate
        actual_cost = cost - entry_fee
        size = actual_cost / entry_price 
        
        return cls(
            side=side,
            entry_price=entry_price,
            entry_date=entry_date,
            size=size,
            entry_fee=entry_fee,
            entry_cashflow=-cost
        )
    
    def close(self, exit_date: str, exit_price: float, fee_rate: float) -> Trade:
        """Close the position and return the completed trade with PnL."""
        gross_proceeds = self.size * exit_price
        exit_fee = gross_proceeds * fee_rate

        if self.side == 'long':
            pnl = self.size * (exit_price - self.entry_price) - exit_fee
        else: 
            pnl = self.size * (self.entry_price - exit_price) - exit_fee

        return Trade(
            entry_date=self.entry_date,
            entry_price=self.entry_price,
            side=self.side,
            size=self.size,
            entry_fee=self.entry_fee,
            entry_cashflow=self.entry_cashflow,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_fee=exit_fee,
            exit_cashflow=-1*self.entry_cashflow + pnl,
            pnl=pnl
        )


class SignalGenerator:
    def __init__(self, metadata_path: str, output_path: str | None = None, prediction_source: str = "prediction_gpt"):
        """
        Initialize signal generator and prepare signals from metadata.
        
        Args:
            metadata_path: Path to the metadata JSON file containing predictions
            output_path: Optional path to save signals CSV file
            prediction_source: Source for predictions - either "prediction_gpt" or "trend"
        """
        self._metadata_path = metadata_path
        self._output_path = output_path
        self._prediction_source = prediction_source
        
        if prediction_source not in ["prediction_gpt", "trend"]:
            raise ValueError("prediction_source must be either 'prediction_gpt' or 'trend'")
            
        self._signals_df = self._prepare_signals()
        
        if output_path:
            self.save_to_csv()
    
    def save_to_csv(self, output_path: str | None = None) -> None:
        """
        Save signals DataFrame to CSV file.
        
        Args:
            output_path: Optional custom path to save the CSV file.
                        If not provided, uses the path specified during initialization.
        """
        save_path = output_path or self._output_path
        if save_path:
            self._signals_df.to_csv(save_path)
            logging.info(f"Saved signals to {save_path}")

    @property
    def signal_data(self) -> pd.DataFrame:
        return self._signals_df

    def _prepare_signals(self) -> pd.DataFrame:
        """
        Prepare DataFrame with trading signals at visible_end_dates.
        Strategy: At each visible_end_date, make a trade decision based on GPT prediction
        for the upcoming hidden period.
        """
        metadata = self._load_metadata()
        rows = []
        last_prediction = None
        
        for plot_name, data in sorted(metadata.items()):
            prediction = data.get(self._prediction_source).lower()
            visible_end_date = data.get('visible_end_date')
            visible_candles = data.get('visible_candles', [])
            
            if not visible_end_date or not visible_candles:
                continue
                
            # Get the last visible candle (at visible_end_date)
            last_visible_candle = visible_candles[-1]
            
            row = {
                'date': pd.to_datetime(visible_end_date),
                'open': last_visible_candle['open'],
                'high': last_visible_candle['high'], 
                'low': last_visible_candle['low'],
                'close': last_visible_candle['close'],
                'volume': last_visible_candle['volume'],
                'plot_name': plot_name,
                'prediction_source': self._prediction_source,
                'prediction': prediction,
                **self._get_default_signals()
            }
            
            # Generate signals based on prediction change
            row.update(self._generate_signals_from_prediction(prediction, last_prediction))
            rows.append(row)
            last_prediction = prediction
            
        df = pd.DataFrame(rows)
        df.set_index('date', inplace=True)
        cols = ['open', 'high', 'low', 'close', 'volume', 'prediction_source', 'prediction',
                'open_long', 'close_long', 'open_short', 'close_short', 'plot_name']
        df = df[cols]
        return df.sort_index()
            
    def _load_metadata(self) -> Dict:
        """Load and validate metadata file."""
        try:
            with open(self._metadata_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Successfully loaded metadata from {self._metadata_path}")
            return metadata
        except Exception as e:
            logging.error(f"Failed to load metadata: {e}")
            raise
            
    def _get_default_signals(self) -> dict:
        """Get the default signal columns and their initial values."""
        return {
            'open_long': False,
            'close_long': False,
            'open_short': False,
            'close_short': False
        }

    def _generate_signals_from_prediction(self, prediction: str, last_prediction: str | None) -> dict:
        """
        Generate trading signals based on current and last prediction.
        
        Args:
            prediction: Current prediction ('bullish', 'bearish' or 'neutral')
            last_prediction: Previous prediction ('bullish', 'bearish' or 'neutral')
        """
        signals = self._get_default_signals()
        
        match (last_prediction, prediction):
            case (None, 'bullish'):
                signals['open_long'] = True
            case (None, 'bearish'):
                signals['open_short'] = True
            
            case ('bullish', 'bearish'):
                signals['close_long'] = True
                signals['open_short'] = True
            case ('bullish', 'neutral'):
                signals['close_long'] = True
                
            case ('bearish', 'bullish'):
                signals['close_short'] = True
                signals['open_long'] = True
            case ('bearish', 'neutral'):
                signals['close_short'] = True
                
            case ('neutral', 'bullish'):
                signals['open_long'] = True
            case ('neutral', 'bearish'):
                signals['open_short'] = True
                
        return signals


class Backtester:
    def __init__(self, signal_generator: SignalGenerator, 
                 initial_capital: float = 10000,
                 position_size: float = 1.,
                 fee_rate: float = 0.001,
                 output_trades_path: str | None = None):
        """
        Initialize backtester with a signal generator and initial capital.
        
        Args:
            signal_generator: Instance of SignalGenerator to provide signals
            initial_capital: Initial capital in quote currency
            position_size: Ratio of capital to use per trade (e.g., 1.0 = 100% of capital)
            fee_rate: Fee rate as decimal (e.g., 0.001 for 0.1%)
            output_trades_path: Optional path to save trade history CSV
        """
        self.signals_df = signal_generator.signal_data
        self.initial_capital = initial_capital
        self._position: Optional[Position] = None
        self._trades: List[Trade] = []
        self._capital = initial_capital
        self._equity: List[List[pd.Timestamp | float]] = []
        self._position_size = position_size
        self._fee_rate = fee_rate
        self._output_trades_path = output_trades_path
    
    @property
    def trades(self) -> List[Trade]:
        """Get list of all completed trades."""
        return self._trades.copy()
    
    @property
    def equity(self) -> pd.Series:
        """Get the equity curve as a time series."""
        dates, values = zip(*self._equity)
        return pd.Series(data=values, index=dates)
    
    def run(self) -> None:
        """Execute trades based on signals."""
        for timestamp, row in self.signals_df.iterrows():
            self._handle_position_close(timestamp, row)
            self._handle_position_open(timestamp, row)
            self._update_equity(timestamp, row['close'])
            
        if self._position is not None:
            last_row = self.signals_df.iloc[-1]
            last_timestamp = self.signals_df.index[-1]
            self._close_position(last_timestamp, last_row['close'])
            
        if self._output_trades_path:
            self._save_trades()
            
    def _handle_position_close(self, timestamp: str, row: pd.Series) -> None:
        """Handle position closing based on signals."""
        if self._position is None:
            return
            
        should_close = (
            (self._position.side == 'long' and row['close_long']) or
            (self._position.side == 'short' and row['close_short'])
        )
        
        if should_close:
            self._close_position(timestamp, row['close'])
            
    def _handle_position_open(self, timestamp: str, row: pd.Series) -> None:
        """Handle position opening based on signals."""
        if self._position is not None:
            return
            
        if row['open_long']:
            self._open_position('long', timestamp, row['close'])
        elif row['open_short']:
            self._open_position('short', timestamp, row['close'])
            
    def _open_position(self, side: str, timestamp: str, price: float) -> None:
        """Open a new position."""
        cost = self._capital * self._position_size
        
        self._position = Position.open(
            side=side,
            entry_price=price,
            entry_date=str(timestamp),
            cost=cost,
            fee_rate=self._fee_rate
        )
        
        self._capital += self._position.entry_cashflow
        
    def _close_position(self, timestamp: str, price: float) -> None:
        """Close current position and record the trade."""
        if self._position is None:
            return
            
        trade = self._position.close(
            exit_date=str(timestamp),
            exit_price=price,
            fee_rate=self._fee_rate
        )
        self._trades.append(trade)
        
        assert trade.exit_cashflow is not None
        self._capital += trade.exit_cashflow
        self._position = None
    
    def _update_equity(self, timestamp: pd.Timestamp, current_price: float) -> None:
        """Update equity and store with timestamp."""
        if self._position is None:
            current_equity = self._capital
        else:
            position_value = self._position.size * current_price
            current_equity = self._capital + position_value + self._calculate_unrealized_pnl(position_value, current_price)
            
        self._equity.append([timestamp, current_equity])
          
    def _calculate_unrealized_pnl(self, position_value: float, current_price: float) -> float:
        """Calculate unrealized P&L for current position at given price."""
        if self._position is None:
            return 0.0

        exit_fee = position_value * self._fee_rate
        
        if self._position.side == 'long':
            return self._position.size * (current_price - self._position.entry_price) - exit_fee
        else: 
            return self._position.size * (self._position.entry_price - current_price) - exit_fee

    def _save_trades(self) -> None:
        """Save trade history to CSV file."""
        trades_data = []
        for trade in self._trades:
            trades_data.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'entry_fee': trade.entry_fee,
                'exit_fee': trade.exit_fee,
                'entry_cashflow': trade.entry_cashflow,
                'exit_cashflow': trade.exit_cashflow,
                'pnl': trade.pnl
            })
        
        df = pd.DataFrame(trades_data)
        df.to_csv(self._output_trades_path, index=False)
        logging.info(f"Saved trade history to {self._output_trades_path}")


class BacktestAnalysis:
    def __init__(self, backtester: 'Backtester'):
        """
        Initialize analysis with backtester instance.
        
        Args:
            backtester: Instance of Backtester after running simulation
        """
        self.backtester = backtester
        self.initial_capital = backtester.initial_capital 
        self.equity_curve = backtester.equity
        self.returns = self.equity_curve.pct_change()
    
    @property
    def roi(self) -> float:
        return (self.equity_curve.iloc[-1] / self.initial_capital - 1) * 100
    
    @property
    def win_rate(self) -> float:
        if not self.backtester.trades:
            return 0.0
        profitable_trades = sum(1 for trade in self.backtester.trades if trade.pnl and trade.pnl > 0)
        return (profitable_trades / len(self.backtester.trades)) * 100
    
    @property
    def max_drawdown(self) -> float:
        rolling_max = self.equity_curve.expanding().max()
        drawdowns = (self.equity_curve - rolling_max) / rolling_max * 100
        return abs(drawdowns.min())
        
        
    def plot_equity_curve(self, save_path: str | None = None) -> None:
        """Plot equity curve over time."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.equity_curve.index, self.equity_curve.values, color='green', linewidth=1.5)

        fig.autofmt_xdate()
        ax.xaxis.set_major_locator(plt.AutoLocator()) 
        
        plt.title('Equity Curve', pad=20)
        plt.xlabel('Time', labelpad=10)
        plt.ylabel('Value in Quote Currency', labelpad=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def print_metrics(self) -> None:
        """Print all computed metrics."""
        print("\nBacktest Metrics:")
        print(f"ROI: {self.roi:.2f}%")
        print(f"Win Rate: {self.win_rate:.2f}%")
        print(f"Max Drawdown: {self.max_drawdown:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Initialize signal generator with metadata file
    signal_generator = SignalGenerator(
        metadata_path="btc_4h_dataset/metadata.json", 
        output_path="results_signals.csv",
        prediction_source="prediction_gpt" 
        )
    
    # Initialize and run backtester
    backtester = Backtester(
        signal_generator=signal_generator,
        initial_capital=1000,
        position_size=1.0,
        fee_rate=0.0006,
        output_trades_path="results_trades.csv"
    )
    
    # Run the backtest
    backtester.run()
    
    # Analyze results
    analysis = BacktestAnalysis(backtester)
    analysis.print_metrics()
    analysis.plot_equity_curve("results_equity_curve.png")

