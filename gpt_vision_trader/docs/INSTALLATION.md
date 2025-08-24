# 📦 Installation Guide

## 🎯 **Quick Installation**

### **Method 1: Package Installation (Recommended)**
```bash
# Clone repository
git clone https://github.com/yourusername/gpt-vision-trader.git
cd gpt-vision-trader

# Install in development mode
pip install -e .
```

### **Method 2: Dependencies Only**
```bash
# Install dependencies directly
pip install -r requirements.txt
```

## 🔧 **System Requirements**

### **Python Version**
- **Python 3.8+** (3.9+ recommended)
- **pip** package manager

### **Operating Systems**
- ✅ **macOS** (tested)
- ✅ **Linux** (Ubuntu, CentOS, etc.)
- ✅ **Windows** (with WSL recommended)

### **Hardware Requirements**
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB free space
- **Network**: Stable internet connection for API calls

## 📋 **Dependencies**

### **Core Dependencies**
```txt
pandas>=2.0.0          # Data processing
numpy>=1.24.0           # Numerical computing
matplotlib>=3.7.0       # Chart plotting
mplfinance>=0.12.0      # Financial charts
requests>=2.31.0        # HTTP requests
openai>=1.0.0           # OpenAI GPT API
python-dotenv>=1.0.0    # Environment variables
```

### **Optional Dependencies**
```txt
# Development tools
pytest>=7.0.0           # Testing framework
pytest-asyncio>=0.21.0  # Async testing
black>=23.0.0           # Code formatting
flake8>=6.0.0           # Code linting

# Enhanced features
rich>=13.0.0            # Beautiful terminal output
aiohttp>=3.8.0          # Async HTTP client

# Freqtrade integration (if using strategy approach)
freqtrade>=2024.1       # Trading framework
```

## 🚀 **Step-by-Step Installation**

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/gpt-vision-trader.git
cd gpt-vision-trader
```

### **2. Create Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### **3. Install Package**
```bash
# Install in development mode (recommended)
pip install -e .

# Or install in regular mode
pip install .
```

### **4. Verify Installation**
```bash
# Check if package is installed
pip list | grep gpt-vision-trader

# Run integration tests
python gpt_vision_trader/tests/test_integration.py

# Test CLI
python -m gpt_vision_trader.scripts.run_live_trading --help
```

## 🔑 **API Keys Setup**

### **OpenAI API Key**
```bash
# Set environment variable (temporary)
export OPENAI_API_KEY="your_openai_api_key_here"

# Or add to shell profile (permanent)
echo 'export OPENAI_API_KEY="your_openai_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### **Create .env File (Optional)**
```bash
# Create .env file in project root
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
FREQTRADE_URL=http://127.0.0.1:8080
FREQTRADE_USERNAME=freqtrade
FREQTRADE_PASSWORD=SuperSecret1!
EOF
```

## 🔧 **Freqtrade Setup**

### **Install Freqtrade**
```bash
# Method 1: Using pip
pip install freqtrade

# Method 2: From source
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
./setup.sh -i
```

### **Configure Freqtrade API**
Add to your Freqtrade `config.json`:
```json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "error",
        "enable_openapi": false,
        "jwt_secret_key": "somethingrandom",
        "CORS_origins": [],
        "username": "freqtrade",
        "password": "SuperSecret1!",
        "ws_token": "sercet_Ws_t0ken"
    },
    "dry_run": true,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "max_open_trades": 1
}
```

### **Start Freqtrade**
```bash
# Start with API enabled
freqtrade trade --config config.json --strategy DefaultStrategy
```

## 🧪 **Verify Installation**

### **1. Run Integration Tests**
```bash
python gpt_vision_trader/tests/test_integration.py
```

Expected output:
```
✅ Configuration: PASSED
✅ Technical Indicators: PASSED  
✅ Data Processor: PASSED
✅ Chart Generator: PASSED
✅ GPT Analyzer: PASSED
✅ Freqtrade API Client: PASSED

🎉 All integration tests passed!
```

### **2. Test CLI Interface**
```bash
# Test help
python -m gpt_vision_trader.scripts.run_live_trading --help

# Test configuration
python -m gpt_vision_trader.scripts.run_live_trading --test
```

### **3. Test API Connections**
```bash
# Test Freqtrade API
curl http://127.0.0.1:8080/api/v1/ping

# Should return: {"status":"pong"}
```

## 🐳 **Docker Installation (Optional)**

### **Using Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  gpt-vision-trader:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - FREQTRADE_URL=http://freqtrade:8080
    volumes:
      - ./gpt_vision_trader/data:/app/gpt_vision_trader/data
    depends_on:
      - freqtrade
  
  freqtrade:
    image: freqtradeorg/freqtrade:stable
    ports:
      - "8080:8080"
    volumes:
      - ./freqtrade_config:/freqtrade/user_data
    command: trade --config /freqtrade/user_data/config.json
```

```bash
# Build and run
docker-compose up -d
```

### **Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Create data directories
RUN mkdir -p gpt_vision_trader/data/{temp_charts,logs,trading_logs}

# Run application
CMD ["python", "-m", "gpt_vision_trader.scripts.run_live_trading"]
```

## 🔧 **Development Installation**

### **For Contributors**
```bash
# Clone with development dependencies
git clone https://github.com/yourusername/gpt-vision-trader.git
cd gpt-vision-trader

# Install with development extras
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### **Development Tools**
```bash
# Code formatting
black gpt_vision_trader/

# Linting
flake8 gpt_vision_trader/

# Type checking
mypy gpt_vision_trader/

# Testing
pytest gpt_vision_trader/tests/
```

## 🚨 **Troubleshooting**

### **Common Installation Issues**

#### **Permission Denied**
```bash
# Use --user flag
pip install --user -e .

# Or use sudo (not recommended)
sudo pip install -e .
```

#### **Package Not Found**
```bash
# Update pip
pip install --upgrade pip

# Clear pip cache
pip cache purge

# Reinstall
pip uninstall gpt-vision-trader
pip install -e .
```

#### **Import Errors**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip install -e .

# Check if package is installed
pip list | grep gpt-vision-trader
```

#### **OpenAI API Issues**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('API key is valid!')
"
```

#### **Freqtrade Connection Issues**
```bash
# Check if Freqtrade is running
ps aux | grep freqtrade

# Test API endpoint
curl -u freqtrade:SuperSecret1! http://127.0.0.1:8080/api/v1/status

# Check firewall/network
netstat -an | grep 8080
```

## 📱 **Platform-Specific Instructions**

### **macOS**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Install package
pip3 install -e .
```

### **Ubuntu/Debian**
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install system dependencies
sudo apt install build-essential

# Install package
pip3 install -e .
```

### **CentOS/RHEL**
```bash
# Install Python
sudo yum install python3 python3-pip

# Install development tools
sudo yum groupinstall "Development Tools"

# Install package
pip3 install -e .
```

### **Windows**
```powershell
# Install Python from python.org
# Download and install Git

# Clone repository
git clone https://github.com/yourusername/gpt-vision-trader.git
cd gpt-vision-trader

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install package
pip install -e .
```

## ✅ **Installation Checklist**

- [ ] Python 3.8+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Package installed (`pip install -e .`)
- [ ] OpenAI API key configured
- [ ] Freqtrade installed and configured
- [ ] Integration tests pass
- [ ] CLI interface works
- [ ] API connections tested

## 🎉 **Next Steps**

After successful installation:

1. **Configure trading parameters** in `gpt_vision_trader/config/settings.py`
2. **Start Freqtrade** with REST API enabled
3. **Run test mode**: `python -m gpt_vision_trader.scripts.run_live_trading --test`
4. **Start live trading**: `python -m gpt_vision_trader.scripts.run_live_trading`

**Congratulations! GPT Vision Trader is now installed and ready to use! 🚀**
