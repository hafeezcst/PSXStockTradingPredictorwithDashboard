import sys
from numpy import nan
sys.modules['numpy'].NaN = nan  # Add NaN as an alias for nan

# Now import pandas_ta after the patch
import pandas_ta 