import sys
import logging
import os

# Basic logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_initialization():
    logger.debug("Starting initialization test")
    
    # Phase 1: Core Python imports
    try:
        logger.debug("Testing core imports...")
        import numpy as np
        import pandas as pd
        logger.debug("Core imports successful")
    except Exception as e:
        logger.error(f"Core imports failed: {str(e)}")
        return False

    # Phase 2: Qt imports  
    try:
        logger.debug("Testing Qt imports...")
        from PyQt6.QtWidgets import QApplication
        logger.debug("Qt imports successful")
    except Exception as e:
        logger.error(f"Qt imports failed: {str(e)}")
        return False

    # Phase 3: Matplotlib setup
    try:
        logger.debug("Testing matplotlib...")
        import matplotlib
        matplotlib.use('QtAgg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        logger.debug("Matplotlib setup successful")
    except Exception as e:
        logger.error(f"Matplotlib setup failed: {str(e)}")
        return False

    # Phase 4: TensorFlow
    try:
        logger.debug("Testing TensorFlow...")
        from tensorflow.keras.models import Sequential
        logger.debug("TensorFlow import successful")
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {str(e)}")
    except Exception as e:
        logger.error(f"TensorFlow failed: {str(e)}")
        return False

    # Phase 5: Full application
    try:
        logger.debug("Testing full application...")
        app = QApplication(sys.argv)
        logger.debug("QApplication created successfully")
        return True
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.debug("Starting test")
    success = test_initialization()
    logger.debug(f"Test completed: {'SUCCESS' if success else 'FAILURE'}")