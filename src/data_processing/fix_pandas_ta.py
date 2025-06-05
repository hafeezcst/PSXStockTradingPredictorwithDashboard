import numpy as np
import sys
from types import ModuleType

# Create a patch for the numpy functions
class NumpyPatch(ModuleType):
    def __init__(self):
        super().__init__('numpy')
        self.NaN = np.nan
        self.log10 = np.log10
        # Copy all numpy attributes
        for attr in dir(np):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(np, attr))

# Apply the patch
sys.modules['numpy'] = NumpyPatch()

# Now import pandas_ta
import pandas_ta as ta

# Restore original numpy
sys.modules['numpy'] = np 