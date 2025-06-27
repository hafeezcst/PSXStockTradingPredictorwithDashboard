import sys
import logging
from PyQt6.QtWidgets import QApplication
from src.data_processing.stock_analysis.ui_manager import MainWindow

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_mainwindow():
    logger.debug("Testing MainWindow initialization")
    app = QApplication(sys.argv)
    
    try:
        logger.debug("Creating MainWindow instance")
        window = MainWindow()
        logger.debug("MainWindow created successfully")
        window.show()
        return app.exec()
    except Exception as e:
        logger.error(f"MainWindow failed: {str(e)}")
        return 1

if __name__ == "__main__":
    logger.debug("Starting MainWindow test")
    sys.exit(test_mainwindow())