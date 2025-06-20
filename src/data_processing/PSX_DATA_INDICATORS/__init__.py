"""
PSX Data Indicators Package

Enhanced technical indicator processing for Pakistan Stock Exchange (PSX) data.
Provides high-performance, modern Python tools for financial data analysis.
"""

from .enhanced_psx_indicator_processor import (
    EnhancedPSXIndicatorProcessor,
    ProcessorConfig,
    DataValidator,
    IndicatorCalculator
)

__version__ = "2.0.0"
__author__ = "PSX Trading Predictor Team"
__email__ = "support@psxtrading.com"

__all__ = [
    "EnhancedPSXIndicatorProcessor",
    "ProcessorConfig", 
    "DataValidator",
    "IndicatorCalculator"
]
