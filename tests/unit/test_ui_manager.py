import pytest
from src.data_processing.stock_analysis.ui_manager import MainWindow
from PyQt6.QtWidgets import QApplication
import sys

def test_mainwindow_instantiation(qtbot):
    """Test MainWindow instantiation and basic UI setup."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "PSX Stock Analysis Dashboard"

def test_dark_mode_toggle(qtbot):
    """Test toggling dark mode changes stylesheet."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    qtbot.addWidget(window)
    initial_style = window.styleSheet()
    window.toggle_dark_mode()
    assert window.styleSheet() != initial_style 