#!/usr/bin/env python3
"""
Setup script for GPT Vision Trader
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "AI-powered trading system using GPT vision analysis of candlestick charts"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "mplfinance>=0.12.0",
        "requests>=2.31.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0"
    ]

setup(
    name="gpt-vision-trader",
    version="1.0.0",
    description="AI-powered trading system using GPT vision analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GPT Vision Trader Team",
    author_email="",
    url="https://github.com/yourusername/gpt-vision-trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "freqtrade": [
            "freqtrade>=2024.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpt-vision-trader=gpt_vision_trader.scripts.run_live_trading:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
