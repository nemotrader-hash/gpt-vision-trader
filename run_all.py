#!/usr/bin/env python3
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from backtest import BacktestAnalysis, Backtester, SignalGenerator
from create_dataset import (
    ATRIndicator,
    MACDIndicator,
    RSIIndicator,
    SMAIndicator,
    create_dataset,
)
from gpt_analysis import PredictionAnalyzer, process_images_and_update_json

# Configuration
SYMBOL = "BTC/USDT"

START_YEAR_MONTH = "2025-03"
END_YEAR_MONTH = "2025-04"
VISIBLE_DAYS = 6
HIDDEN_DAYS = 1
# Change this line for significant cost savings
GPT_MODEL = "gpt-4.1"  # Instead of "gpt-4.1"
# GPT_MODEL = "gpt-4o-mini"


TIMEFRAME = "15m"
FOLDER_NAME = f"test_{SYMBOL.replace('/', '_').lower()}_{TIMEFRAME}_{START_YEAR_MONTH}_{END_YEAR_MONTH}_{GPT_MODEL}_3_days_sma"

# Execution flags
DO_DATASET = True  
DO_GPT_ANALYSIS = True  
DO_PREDICTION_ANALYSIS = True 
DO_BACKTEST = True      

# Trading parameters
INITIAL_CAPITAL = 1000  
POSITION_SIZE = 1.0    
FEE_RATE = 0.0006      

# Prediction source for backtesting
PREDICTION_SOURCE = "prediction_gpt"  # or "trend" to use actual trend outcomes for comparison

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def setup_directories():
    """Create necessary directories."""
    test_dir = Path(FOLDER_NAME)
    test_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = test_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    dataset_dir_plots = dataset_dir / "plots"
    dataset_dir_plots.mkdir(exist_ok=True)

def get_technical_indicators():
    """Configure and return technical indicators."""
    return {
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
        # "RSI14": RSIIndicator(
        #     period=14,
        #     color='#8757ff', 
        #     alpha=0.8,
        #     linewidth=1.0,
        #     overbought=70,
        #     oversold=30
        # ),
        # "MACD": MACDIndicator(
        #     fast_period=12,
        #     slow_period=26,
        #     signal_period=9,
        #     color='#2ecc71', 
        #     signal_color='#e74c3c',  
        #     macd_color='#3498db',
        #     alpha=0.8,
        #     linewidth=1.0
        # ),
        # "ATR14": ATRIndicator(
        #     period=14,
        #     color='#e67e22', 
        #     alpha=0.8,
        #     linewidth=1.0
        # )
    }

def save_combined_results(prediction_analysis: PredictionAnalyzer, backtest_analysis: BacktestAnalysis):
    """Save combined results from both analyses to a single file."""
    output_path = os.path.join(FOLDER_NAME, "combined_results.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Write timestamp
        f.write(f"Analysis Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write backtest analysis first
        f.write("Backtest Results\n")
        f.write("-" * 30 + "\n")
        f.write(f"Initial Capital: {INITIAL_CAPITAL} USDT\n")
        f.write(f"Position Size: {POSITION_SIZE*100}%\n")
        f.write(f"Fee Rate: {FEE_RATE*100}%\n\n")
        f.write(f"ROI: {backtest_analysis.roi:.2f}%\n")
        f.write(f"Win Rate: {backtest_analysis.win_rate:.2f}%\n")
        f.write(f"Max Drawdown: {backtest_analysis.max_drawdown:.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # Write prediction analysis
        f.write("\nGPT Prediction Analysis\n")
        f.write("-" * 30 + "\n")
        analysis = prediction_analysis.analyze()
        
        f.write(f"Total Samples: {analysis['total_samples']}\n")
        f.write(f"Overall Accuracy: {analysis['accuracy']:.2%}\n\n")
        
        f.write("Distribution of Predictions:\n")
        for pred, count in analysis['gpt_predictions'].items():
            percentage = count / analysis['total_samples'] * 100
            f.write(f"  {pred.capitalize()}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nDistribution of Actual Trends:\n")
        for trend, count in analysis['actual_trends'].items():
            percentage = count / analysis['total_samples'] * 100
            f.write(f"  {trend.capitalize()}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write("Actual \\ Predicted | Bullish | Bearish | Neutral\n")
        f.write("-" * 45 + "\n")
        for actual in ["bullish", "bearish", "neutral"]:
            pred_counts = analysis['confusion_matrix'][actual]
            f.write(f"{actual.capitalize():15} | {pred_counts['bullish']:7} | {pred_counts['bearish']:7} | {pred_counts['neutral']:7}\n")
        
        f.write("\nDetailed Metrics Analysis:\n")
        for trend, metrics in analysis['class_metrics'].items():
            true_pos = analysis['confusion_matrix'][trend][trend]
            false_pos = sum(analysis['confusion_matrix'][t][trend] for t in ["bullish", "bearish", "neutral"]) - true_pos
            false_neg = sum(analysis['confusion_matrix'][trend][p] for p in ["bullish", "bearish", "neutral"]) - true_pos
            
            f.write(f"\n{trend.capitalize()}:\n")
            f.write(f"  Precision: {metrics['precision']:.2%}\n")
            f.write(f"    - Out of {true_pos + false_pos} {trend} predictions, {true_pos} were correct\n")
            f.write(f"    - When GPT predicts {trend}, it's right {metrics['precision']:.2%} of the time\n")
            
            f.write(f"  Recall: {metrics['recall']:.2%}\n")
            f.write(f"    - Out of {true_pos + false_neg} actual {trend} cases, caught {true_pos}\n")
            f.write(f"    - GPT catches {metrics['recall']:.2%} of all {trend} trends\n")
            
            f.write(f"  F1 Score: {metrics['f1']:.2%}\n")

async def main():
    """Main execution function."""
    setup_logging()
    setup_directories()
    
    metadata_path = os.path.join(FOLDER_NAME, "dataset", "metadata.json")
    images_dir = os.path.join(FOLDER_NAME, "dataset", "plots")
    
    if DO_DATASET:
        logging.info("Step 1: Creating dataset...")
        create_dataset(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_year_month=START_YEAR_MONTH,
            end_year_month=END_YEAR_MONTH,
            visible_days=VISIBLE_DAYS,
            hidden_days=HIDDEN_DAYS,
            output_dir=os.path.join(FOLDER_NAME, "dataset"),
            technical_indicators=get_technical_indicators()
        )
    
    if DO_GPT_ANALYSIS:
        logging.info("Step 2: Running GPT analysis...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set")
        
        metadata_paths = [
            os.path.join(FOLDER_NAME, "dataset", "metadata.json"),
            os.path.join(FOLDER_NAME, "dataset", "metadata2.json")
        ]
        await process_images_and_update_json(metadata_paths, images_dir, api_key, use_batch=True, gpt_model=GPT_MODEL)
       
    if DO_BACKTEST:
        logging.info("Step 3: Running backtest...")
        signal_generator = SignalGenerator(
            metadata_path=metadata_path,
            output_path=os.path.join(FOLDER_NAME, "signals.csv"),
            prediction_source=PREDICTION_SOURCE
        )
        
        backtester = Backtester(
            signal_generator=signal_generator,
            initial_capital=INITIAL_CAPITAL,
            position_size=POSITION_SIZE,
            fee_rate=FEE_RATE,
            output_trades_path=os.path.join(FOLDER_NAME, "trades.csv")
        )
        
        backtester.run()
        
        logging.info("Step 4: Analyzing and saving results...")
        backtest_analysis = BacktestAnalysis(backtester)
        backtest_analysis.plot_equity_curve(os.path.join(FOLDER_NAME, "equity_curve.png"))
        
        prediction_analyzer = PredictionAnalyzer(metadata_path)
        save_combined_results(prediction_analyzer, backtest_analysis)
    
    logging.info(f"All results have been saved to the {FOLDER_NAME} directory")
    logging.info("Script execution completed successfully")

if __name__ == "__main__":
    asyncio.run(main()) 