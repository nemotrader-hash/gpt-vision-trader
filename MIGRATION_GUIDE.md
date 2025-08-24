# 📦 Migration Guide - Old to New Structure

## 🎯 **Quick Migration**

The repository has been completely restructured. Here's how to migrate from the old `live_tools/` approach:

### **❌ Old Way (Deprecated)**
```bash
# Old command
python live_tools/run_api_trading.py --test

# Old imports
from live_tools.config import LiveTradingConfig
from live_tools.freqtrade_api_client import FreqtradeAPIClient
```

### **✅ New Way (Current)**
```bash
# New command
python -m gpt_vision_trader.scripts.run_live_trading --test

# New imports
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient
```

## 🔄 **File Mapping**

| Old File | New Location | Status |
|----------|-------------|---------|
| `live_tools/config.py` | `gpt_vision_trader/config/settings.py` | ✅ Improved |
| `live_tools/freqtrade_api_client.py` | `gpt_vision_trader/api/freqtrade_client.py` | ✅ Simplified |
| `live_tools/api_trading_controller.py` | `gpt_vision_trader/api/trading_controller.py` | ✅ Streamlined |
| `live_tools/live_gpt_analyzer.py` | `gpt_vision_trader/core/gpt_analyzer.py` | ✅ Enhanced |
| `live_tools/run_api_trading.py` | `gpt_vision_trader/scripts/run_live_trading.py` | ✅ Professional CLI |
| `create_dataset.py` | `gpt_vision_trader/core/chart_generator.py` + `gpt_vision_trader/core/data_processor.py` | ✅ Modular |
| `gpt_analysis.py` | `gpt_vision_trader/core/gpt_analyzer.py` | ✅ Improved |

## 🚀 **Key Changes**

### **1. Simplified Trade Execution**
```python
# ❌ Old: Complex stake calculations
stake_amount = self._calculate_stake_amount()
response = self.api.force_enter(pair=pair, stake_amount=stake_amount)

# ✅ New: Let Freqtrade handle it
response = self.api.force_enter(pair=pair, side='long')
```

### **2. Clean Configuration**
```python
# ❌ Old: Multiple config classes
from live_tools.config import LiveTradingConfig
from live_tools.freqtrade_api_client import FreqtradeConfig

# ✅ New: Centralized configuration
from gpt_vision_trader.config.settings import TradingConfig
from gpt_vision_trader.api.freqtrade_client import FreqtradeConfig
```

### **3. Professional Package Structure**
```python
# ❌ Old: Direct imports
from live_tools.live_gpt_analyzer import LiveGPTAnalyzer

# ✅ New: Package imports
from gpt_vision_trader.core.gpt_analyzer import LiveGPTAnalyzer
from gpt_vision_trader import GPTAnalyzer  # Also available from main package
```

### **4. Modern CLI Interface**
```bash
# ❌ Old: Script-based
python live_tools/run_api_trading.py --symbol BTC/USDT --test

# ✅ New: Module-based with profiles
python -m gpt_vision_trader.scripts.run_live_trading --symbol BTC/USDT --test --profile dev
```

## 📋 **Migration Checklist**

### **For Existing Users:**
- [ ] Update import statements to new package structure
- [ ] Change command-line usage to new module format
- [ ] Update configuration to use new `TradingConfig` class
- [ ] Remove old `live_tools/` references
- [ ] Test with new CLI: `python -m gpt_vision_trader.scripts.run_live_trading --test`

### **For Developers:**
- [ ] Install package: `pip install -e .`
- [ ] Update imports in custom scripts
- [ ] Use new configuration profiles (dev/prod)
- [ ] Migrate any custom extensions to new structure

## 🔧 **Installation**

### **Old Way**
```bash
# Manual dependency management
pip install -r live_tools/requirements_live.txt
python live_tools/setup_live_trading.sh
```

### **New Way**
```bash
# Proper package installation
pip install -e .

# Or just dependencies
pip install -r requirements.txt
```

## 📊 **Benefits After Migration**

### **Simplified Trading**
- ✅ No more stake amount calculations
- ✅ Clean `forceenter`/`forceexit` calls
- ✅ Full position exits only
- ✅ Freqtrade handles position sizing

### **Better Organization**
- ✅ Professional package structure
- ✅ Modular components (core, api, utils, config)
- ✅ Easy imports and extensions
- ✅ Proper Python package

### **Enhanced Configuration**
- ✅ Multiple profiles (dev, prod, default)
- ✅ Centralized settings
- ✅ Environment variable integration
- ✅ Configuration validation

### **Improved CLI**
- ✅ Professional command-line interface
- ✅ Multiple profiles and options
- ✅ Better help and documentation
- ✅ Consistent argument handling

## 🚨 **Breaking Changes**

### **Import Paths**
All import paths have changed. Update your imports:
```python
# Old → New
from live_tools.config import LiveTradingConfig
# becomes:
from gpt_vision_trader.config.settings import TradingConfig

from live_tools.freqtrade_api_client import FreqtradeAPIClient  
# becomes:
from gpt_vision_trader.api.freqtrade_client import FreqtradeAPIClient
```

### **Configuration Classes**
```python
# Old
config = LiveTradingConfig.from_run_all_settings()

# New
config = TradingConfig.from_run_all_settings()
```

### **Command Line**
```bash
# Old
python live_tools/run_api_trading.py

# New  
python -m gpt_vision_trader.scripts.run_live_trading
```

## ✅ **Compatibility Maintained**

### **Trading Logic**
- ✅ Same GPT analysis pipeline
- ✅ Identical chart generation
- ✅ Same technical indicators
- ✅ Same prediction accuracy

### **Freqtrade Integration**
- ✅ Same API endpoints
- ✅ Same authentication
- ✅ Same trade execution (simplified)
- ✅ Same monitoring capabilities

### **Results**
- ✅ Same backtesting compatibility
- ✅ Same performance expectations
- ✅ Same risk management
- ✅ Same profit potential

---

## 🎉 **Ready to Migrate!**

The new structure is **production-ready** and provides significant improvements while maintaining full compatibility with your existing trading strategies.

**Start your migration today:**
1. `pip install -e .`
2. `python -m gpt_vision_trader.scripts.run_live_trading --test`
3. Update your imports and commands
4. Enjoy the cleaner, more professional system! 🚀
