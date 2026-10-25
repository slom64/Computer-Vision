"""
Main Application Entry Point:
Sensum PATVIS APA — Fluid Bed In-Line Agglomeration & Sticking Monitor.
Compatible with Daheng MER2-301-125U3M USB 3.0 Industrial Monochrome Camera.
"""

import sys
import os

# Ensure current directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from fluid_bed_pat.main_window import PatvisMainWindow


def main():
    # Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Sensum PATVIS APA - Fluid Bed Agglomeration Monitor")
    app.setOrganizationName("FluidBed Industrial Vision")

    # Set default clean system font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = PatvisMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
