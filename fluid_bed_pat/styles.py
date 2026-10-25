"""
Sensum PATVIS APA High-End Industrial Dark Theme Stylesheet.
Engineered for clarity, readability, and non-technical factory operators.
"""

INDUSTRIAL_DARK_STYLESHEET = """
QMainWindow {
    background-color: #0f1219;
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

QWidget {
    background-color: transparent;
    color: #e2e8f0;
    font-size: 12px;
}

QToolTip {
    background-color: #1e2638;
    color: #ffffff;
    border: 1px solid #3b4866;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 11px;
}

QGroupBox {
    border: 1px solid #242e42;
    border-radius: 8px;
    margin-top: 18px;
    font-weight: 700;
    color: #8fa0b5;
    background-color: #141923;
    padding: 12px 10px 10px 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    left: 12px;
    color: #00d2ff;
    font-size: 10px;
    letter-spacing: 0.8px;
    background-color: #1a2233;
    border-radius: 4px;
}

QLabel {
    color: #cbd5e1;
}

/* Modern tactile buttons */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #252f44, stop:1 #1c2333);
    color: #ffffff;
    border: 1px solid #33415c;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 12px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #303d59, stop:1 #242e42);
    border-color: #00d2ff;
}

QPushButton:pressed {
    background: #141a26;
    border-color: #0099cc;
}

QPushButton:disabled {
    background: #141822;
    color: #475569;
    border-color: #1e2535;
}

/* Primary Big Action Buttons for Operators */
QPushButton#btn_start {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00b074, stop:1 #007a50);
    color: #ffffff;
    border: 1px solid #00d68f;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.5px;
    border-radius: 6px;
    padding: 8px 18px;
}
QPushButton#btn_start:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00cc86, stop:1 #008f5e);
}

QPushButton#btn_stop {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e03131, stop:1 #a61e1e);
    color: #ffffff;
    border: 1px solid #ff4d4d;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.5px;
    border-radius: 6px;
    padding: 8px 18px;
}
QPushButton#btn_stop:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff4d4d, stop:1 #c92a2a);
}

QPushButton#btn_report {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #253858, stop:1 #172b4d);
    color: #00d2ff;
    border: 1px solid #00d2ff;
    font-weight: 700;
    padding: 7px 14px;
}
QPushButton#btn_report:hover {
    background: #00d2ff;
    color: #0c1a2e;
}

QComboBox {
    background-color: #171d2a;
    color: #ffffff;
    border: 1px solid #2e3a52;
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 26px;
    font-size: 12px;
}

QComboBox:hover {
    border-color: #00d2ff;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #2e3a52;
}

QComboBox QAbstractItemView {
    background-color: #141924;
    color: #ffffff;
    selection-background-color: #24334a;
    selection-color: #00d2ff;
    border: 1px solid #2e3a52;
    padding: 4px;
    outline: none;
}

QSlider::groove:horizontal {
    height: 5px;
    background: #202838;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00a3ff, stop:1 #00d2ff);
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #00d2ff;
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #00d2ff;
}

QCheckBox {
    color: #cbd5e1;
    spacing: 8px;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 17px;
    height: 17px;
    border: 1.5px solid #33415c;
    border-radius: 4px;
    background-color: #171d2a;
}

QCheckBox::indicator:checked {
    background-color: #00d2ff;
    border-color: #00d2ff;
}

QTabWidget::pane {
    border: 1px solid #242e42;
    background: #121620;
    border-radius: 6px;
}

QTabBar::tab {
    background: #181e2b;
    color: #8fa0b5;
    padding: 7px 16px;
    margin-right: 3px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background: #242e42;
    color: #00d2ff;
    border-bottom: 2px solid #00d2ff;
}

QTabBar::tab:hover {
    color: #ffffff;
}

QScrollBar:vertical {
    background: #11141c;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #242e42;
    min-height: 24px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #00d2ff;
}

QScrollBar:horizontal {
    background: #11141c;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #242e42;
    min-width: 24px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #00d2ff;
}

QStatusBar {
    background-color: #0a0d13;
    color: #8fa0b5;
    border-top: 1px solid #1a2233;
    font-size: 11px;
    padding: 3px;
}
"""
