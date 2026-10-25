import os
import sys
import time
import webbrowser
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QFrame, QScrollArea, QTabWidget,
    QProgressBar, QSizePolicy, QLineEdit, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QIcon

from fluid_bed_pat.camera_manager import IndustrialCameraManager, CameraDeviceInfo
from fluid_bed_pat.analytics_engine import PATAnalyticsEngine, ProcessAnalyticsSnapshot, AgglomerateSample
from fluid_bed_pat.inference_pipeline import InferencePipeline, InspectionWorker, HardwareAcceleratorManager
from fluid_bed_pat.charts_widget import PATTrendChartsWidget
from fluid_bed_pat.sample_gallery_widget import AgglomerateGalleryWidget
from fluid_bed_pat.widgets import CircularProcessGauge, OperatorStatusBanner, ModernOperatorCard, BatchReportGenerator
from fluid_bed_pat.styles import INDUSTRIAL_DARK_STYLESHEET


class LiveVideoWidget(QLabel):
    """
    High-performance video canvas that preserves aspect ratio and renders
    fluid bed inspection overlays cleanly.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(540, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._current_pixmap: Optional[QPixmap] = None
        self.setText("NO ACTIVE CAMERA FEED\n\n1. Select a camera or dataset above\n2. Click 'Connect'\n3. Click '▶ START INSPECTION'")
        self.setStyleSheet("""
            QLabel {
                background-color: #0b0d13;
                color: #64748b;
                font-size: 13px;
                font-weight: 600;
                border: 1.5px solid #1f2738;
                border-radius: 8px;
                text-align: center;
            }
        """)

    def update_frame(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self):
        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            scaled = self._current_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class PatvisMainWindow(QMainWindow):
    """
    Sensum PATVIS APA-Style Fluid Bed In-Line Agglomeration & Sticking Monitor.
    Engineered with a streamlined, non-technical Operator View and an optional Engineer View.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensum PATVIS APA — Fluid Bed In-Line Agglomeration & Sticking Monitor")
        self.resize(1620, 960)
        self.setStyleSheet(INDUSTRIAL_DARK_STYLESHEET)

        # Core subsystems
        self.camera_manager = IndustrialCameraManager()
        self.pipeline = InferencePipeline(
            yolo_model_path="outputs/models/pellet_detector_best.pt",
            seq_model_path=None,  # High-FPS real-time mode
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            conf_thresh=0.25,
            iou_thresh=0.45,
            imgsz=1024,
            wall_sticking_enabled=True
        )
        self.worker: Optional[InspectionWorker] = None
        self._last_snapshot: Optional[ProcessAnalyticsSnapshot] = None
        self.batch_id: str = f"BATCH-{int(time.time()) % 100000:05d}"

        # Build UI layout
        self._setup_ui()

        # Initial device scan
        self.refresh_camera_devices()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. Top Modern Header Bar
        main_layout.addLayout(self._create_header_bar())

        # 2. Operator Illuminated Status Banner
        self.status_banner = OperatorStatusBanner()
        main_layout.addWidget(self.status_banner)

        # 3. Main Workspace Splitter (Left: Controls & KPIs, Right: Video + Charts & Gallery)
        workspace_splitter = QSplitter(Qt.Horizontal)
        workspace_splitter.setHandleWidth(6)

        # Left Panel (KPI Cards & Simplified / Advanced Settings)
        left_widget = self._create_left_panel()
        workspace_splitter.addWidget(left_widget)
        workspace_splitter.setStretchFactor(0, 0)  # Fixed width

        # Right Panel (Viewport + Bottom Tabbed Section)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(6)

        # Viewport with Overlay Controls
        viewport_container = self._create_viewport_container()
        right_splitter.addWidget(viewport_container)
        right_splitter.setStretchFactor(0, 3)

        # Bottom Section (Charts & Defect Sample Gallery)
        bottom_tabs = self._create_bottom_section()
        right_splitter.addWidget(bottom_tabs)
        right_splitter.setStretchFactor(1, 2)

        workspace_splitter.addWidget(right_splitter)
        workspace_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(workspace_splitter)

        # Status Bar
        self.statusBar().showMessage("Ready. Select your camera or test dataset to begin.")

    def _create_header_bar(self) -> QHBoxLayout:
        hdr = QHBoxLayout()
        hdr.setSpacing(12)

        # Sensum Branding
        brand_box = QVBoxLayout()
        brand_title = QLabel("SENSUM PATVIS APA")
        brand_title.setStyleSheet("font-size: 17px; font-weight: 900; color: #00d2ff; letter-spacing: 1.2px;")
        sub_title = QLabel("FLUID BED IN-LINE AGGLOMERATION & STICKING MONITOR")
        sub_title.setStyleSheet("font-size: 9px; font-weight: 700; color: #64748b; letter-spacing: 0.6px;")
        brand_box.addWidget(brand_title)
        brand_box.addWidget(sub_title)
        hdr.addLayout(brand_box)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #242e42; max-width: 1px;")
        hdr.addWidget(sep)

        # Friendly Camera Selector
        cam_box = QHBoxLayout()
        cam_icon = QLabel("📷")
        cam_icon.setStyleSheet("font-size: 14px;")
        cam_box.addWidget(cam_icon)

        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(320)
        self.cam_combo.setToolTip("Choose your connected industrial camera or recorded fluid bed dataset")
        cam_box.addWidget(self.cam_combo)

        self.refresh_cam_btn = QPushButton("🔄 Refresh")
        self.refresh_cam_btn.setToolTip("Re-scan for connected USB3 and DirectShow cameras")
        self.refresh_cam_btn.clicked.connect(self.refresh_camera_devices)
        cam_box.addWidget(self.refresh_cam_btn)

        self.connect_cam_btn = QPushButton("Connect Camera")
        self.connect_cam_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1f3a5f, stop:1 #13243d);
                color: #00d2ff;
                border: 1.5px solid #00d2ff;
                font-weight: 700;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover { background: #00d2ff; color: #0c1a2e; }
        """)
        self.connect_cam_btn.clicked.connect(self.toggle_camera_connection)
        cam_box.addWidget(self.connect_cam_btn)

        hdr.addLayout(cam_box)

        # Tactile Action Buttons (Start, Stop, Report)
        self.start_btn = QPushButton("▶  START INSPECTION")
        self.start_btn.setObjectName("btn_start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_inspection)
        hdr.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹  STOP")
        self.stop_btn.setObjectName("btn_stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_inspection)
        hdr.addWidget(self.stop_btn)

        self.report_btn = QPushButton("📄  BATCH REPORT")
        self.report_btn.setObjectName("btn_report")
        self.report_btn.setToolTip("Export printable QA batch report with defect gallery")
        self.report_btn.clicked.connect(self.export_batch_report)
        hdr.addWidget(self.report_btn)

        hdr.addStretch()

        # FPS & Hardware badges
        hw_box = QVBoxLayout()
        self.fps_lbl = QLabel("Camera: 0.0 FPS  |  AI: 0.0 FPS")
        self.fps_lbl.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        
        self.hw_lbl = QLabel("Hardware: Auto-detecting GPU...")
        self.hw_lbl.setStyleSheet("color: #00e676; font-size: 10px; font-weight: bold;")
        hw_box.addWidget(self.fps_lbl)
        hw_box.addWidget(self.hw_lbl)
        hdr.addLayout(hw_box)

        self._update_hardware_hud()

        return hdr

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Scroll Area for left panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 4, 0)
        c_layout.setSpacing(10)

        # ---------------- 1. GAUGES & PROCESS SUMMARY CARD ----------------
        gauge_box = QFrame()
        gauge_box.setStyleSheet("""
            QFrame {
                background-color: #141824;
                border: 1px solid #242e42;
                border-radius: 8px;
            }
        """)
        gb_layout = QVBoxLayout(gauge_box)
        gb_layout.setContentsMargins(12, 10, 12, 10)
        gb_layout.setSpacing(6)

        gb_title = QLabel("PROCESS AGGLOMERATION METER (PAT)")
        gb_title.setStyleSheet("font-size: 10px; font-weight: 800; color: #00d2ff; letter-spacing: 0.8px;")
        gb_layout.addWidget(gb_title)

        center_gauge_row = QHBoxLayout()
        center_gauge_row.addStretch()
        self.circular_gauge = CircularProcessGauge()
        center_gauge_row.addWidget(self.circular_gauge)
        center_gauge_row.addStretch()
        gb_layout.addLayout(center_gauge_row)

        self.gauge_sub_lbl = QLabel("0.0% Smoothed Rate • Stable Fluidization")
        self.gauge_sub_lbl.setAlignment(Qt.AlignCenter)
        self.gauge_sub_lbl.setStyleSheet("color: #8fa0b5; font-size: 11px; font-weight: 600;")
        gb_layout.addWidget(self.gauge_sub_lbl)

        c_layout.addWidget(gauge_box)

        # ---------------- 2. MODERN OPERATOR CARDS ----------------
        self.card_active = ModernOperatorCard("🧪", "Active Flowing Particles")
        self.card_active.set_content("0 Singles  |  0 Agglom", "Raw Frame Count: 0 Total", "#00d2ff")
        c_layout.addWidget(self.card_active)

        self.card_cumulative = ModernOperatorCard("📦", "Total Pellets Inspected")
        self.card_cumulative.set_content("0 Pellets", "0 Unique Singles • 0 Agglomerates (0.0%)", "#00e676")
        c_layout.addWidget(self.card_cumulative)

        self.card_stuck = ModernOperatorCard("🧱", "Chamber Wall Adhesion")
        self.card_stuck.set_content("NORMAL: Clear", "0 Sticking Warnings • 0 Wall Caking", "#00e676")
        c_layout.addWidget(self.card_stuck)

        self.card_rate = ModernOperatorCard("⚡", "Inspection Throughput")
        self.card_rate.set_content("0.0 Pellets / sec", "Real-time process speed", "#ffffff")
        c_layout.addWidget(self.card_rate)

        # ---------------- 3. MODE TOGGLE & SETTINGS DRAWER ----------------
        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Settings View:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["👤 Operator Mode (Simplified)", "⚙️ Engineer Mode (Advanced Tuning)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_switched)
        mode_box.addWidget(self.mode_combo)
        c_layout.addLayout(mode_box)

        # Stacked Widget for Operator vs Engineer settings
        self.settings_stack = QStackedWidget()

        # Page 0: Operator Simplified Controls
        op_page = QWidget()
        op_layout = QVBoxLayout(op_page)
        op_layout.setContentsMargins(0, 0, 0, 0)
        op_layout.setSpacing(8)

        op_grp = QGroupBox("OPERATOR PROCESS TUNING")
        og_layout = QVBoxLayout(op_grp)
        og_layout.setSpacing(8)

        og_layout.addWidget(QLabel("Alert Sensitivity Level:"))
        self.sens_combo = QComboBox()
        self.sens_combo.addItems(["🟢 Relaxed (Warning > 15%, Alarm > 30%)", "🟡 Standard (Warning > 12%, Alarm > 25%)", "🔴 Strict (Warning > 8%, Alarm > 18%)"])
        self.sens_combo.setCurrentIndex(1)
        self.sens_combo.currentIndexChanged.connect(self._on_sensitivity_preset_changed)
        og_layout.addWidget(self.sens_combo)

        og_layout.addWidget(QLabel("Camera Brightness (Exposure):"))
        self.op_exp_slider = QSlider(Qt.Horizontal)
        self.op_exp_slider.setRange(1000, 25000)
        self.op_exp_slider.setValue(10000)
        self.op_exp_slider.valueChanged.connect(lambda val: self.camera_manager.set_exposure_us(float(val)))
        og_layout.addWidget(self.op_exp_slider)

        reset_btn = QPushButton("🔄 Reset Batch Counters")
        reset_btn.setStyleSheet("""
            QPushButton {
                background: #1e2636;
                color: #ffaa00;
                border: 1px solid #ffaa00;
                font-weight: 700;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:hover { background: #ffaa00; color: #000; }
        """)
        reset_btn.clicked.connect(self.reset_batch)
        og_layout.addWidget(reset_btn)

        op_layout.addWidget(op_grp)
        self.settings_stack.addWidget(op_page)

        # Page 1: Engineer Technical Tuning
        eng_page = QWidget()
        eng_layout = QVBoxLayout(eng_page)
        eng_layout.setContentsMargins(0, 0, 0, 0)
        eng_layout.setSpacing(6)

        eng_grp = QGroupBox("ENGINEER / SCIENTIST TUNING")
        eg_layout = QVBoxLayout(eng_grp)
        eg_layout.setSpacing(6)

        eg_layout.addWidget(QLabel("Smoothing Factor (EMA α):"))
        self.ema_slider = QSlider(Qt.Horizontal)
        self.ema_slider.setRange(2, 50)
        self.ema_slider.setValue(15)
        self.ema_lbl = QLabel("0.15 (Default PAT)")
        self.ema_lbl.setStyleSheet("color: #00d2ff; font-size: 10px;")
        self.ema_slider.valueChanged.connect(self._on_ema_slider_changed)
        eg_layout.addWidget(self.ema_slider)
        eg_layout.addWidget(self.ema_lbl)

        eg_layout.addWidget(QLabel("Rolling Window Duration:"))
        self.roll_slider = QSlider(Qt.Horizontal)
        self.roll_slider.setRange(1, 10)
        self.roll_slider.setValue(3)
        self.roll_lbl = QLabel("3.0 Seconds")
        self.roll_lbl.setStyleSheet("color: #00d2ff; font-size: 10px;")
        self.roll_slider.valueChanged.connect(self._on_roll_slider_changed)
        eg_layout.addWidget(self.roll_slider)
        eg_layout.addWidget(self.roll_lbl)

        eg_layout.addWidget(QLabel("Detection Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(25)
        self.conf_lbl = QLabel("0.25")
        self.conf_lbl.setStyleSheet("color: #00d2ff; font-size: 10px;")
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        eg_layout.addWidget(self.conf_slider)
        eg_layout.addWidget(self.conf_lbl)

        eg_layout.addWidget(QLabel("Compute Hardware Accelerator:"))
        self.gpu_combo = QComboBox()
        devices = HardwareAcceleratorManager.get_available_devices()
        for d in devices:
            self.gpu_combo.addItem(d['label'], d['device_str'])
        for i in range(self.gpu_combo.count()):
            if self.gpu_combo.itemData(i) == self.pipeline.device:
                self.gpu_combo.setCurrentIndex(i)
                break
        self.gpu_combo.currentIndexChanged.connect(self._on_gpu_device_changed)
        eg_layout.addWidget(self.gpu_combo)

        self.cb_wall_monitor = QCheckBox("Chamber Wall Sticking Monitor")
        self.cb_wall_monitor.setChecked(True)
        self.cb_wall_monitor.toggled.connect(lambda val: setattr(self.pipeline, 'wall_sticking_enabled', val))
        eg_layout.addWidget(self.cb_wall_monitor)

        eng_layout.addWidget(eng_grp)
        self.settings_stack.addWidget(eng_page)

        c_layout.addWidget(self.settings_stack)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        return panel

    def _create_viewport_container(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Toolbar above viewport
        tb = QHBoxLayout()
        v_title = QLabel("INSPECTION LIVE VIEWPORT")
        v_title.setStyleSheet("font-weight: 800; color: #00d2ff; font-size: 11px; letter-spacing: 0.8px;")
        tb.addWidget(v_title)

        tb.addStretch()

        self.cb_boxes = QCheckBox("Boxes")
        self.cb_boxes.setChecked(True)
        self.cb_boxes.toggled.connect(lambda val: setattr(self.pipeline, 'show_boxes', val))
        tb.addWidget(self.cb_boxes)

        self.cb_labels = QCheckBox("Labels")
        self.cb_labels.setChecked(True)
        self.cb_labels.toggled.connect(lambda val: setattr(self.pipeline, 'show_labels', val))
        tb.addWidget(self.cb_labels)

        self.cb_trails = QCheckBox("Flow Trails")
        self.cb_trails.setChecked(True)
        self.cb_trails.toggled.connect(lambda val: setattr(self.pipeline, 'show_trails', val))
        tb.addWidget(self.cb_trails)

        self.cb_walls = QCheckBox("Wall Zones")
        self.cb_walls.setChecked(True)
        self.cb_walls.toggled.connect(lambda val: setattr(self.pipeline, 'show_wall_zones', val))
        tb.addWidget(self.cb_walls)

        snap_btn = QPushButton("📸  Snapshot")
        snap_btn.setToolTip("Save high-resolution image of the current inspection frame")
        snap_btn.clicked.connect(self.save_snapshot)
        tb.addWidget(snap_btn)

        layout.addLayout(tb)

        # Main video widget
        self.video_widget = LiveVideoWidget()
        layout.addWidget(self.video_widget, stretch=1)

        return container

    def _create_bottom_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        bottom_tabs = QTabWidget()

        # 1. Defect & Agglomerate Gallery (Sensum PATVIS Feature)
        self.gallery_widget = AgglomerateGalleryWidget()
        bottom_tabs.addTab(self.gallery_widget, "🚨 Live Captured Agglomerate Samples (Click to Zoom)")

        # 2. PAT Trend Charts
        self.charts_widget = PATTrendChartsWidget(self.pipeline.analytics_engine)
        bottom_tabs.addTab(self.charts_widget, "📈 Process Analytical Trend Charts (Agglomeration & Counts)")

        layout.addWidget(bottom_tabs)
        return container

    # ---------------- CAMERA MANAGEMENT ----------------
    def refresh_camera_devices(self):
        """Scans for available hardware cameras and dataset feeds with user-friendly names."""
        self.cam_combo.clear()
        devices = IndustrialCameraManager.enumerate_cameras()
        self._available_devices = devices

        for d in devices:
            # Human-friendly labels
            if d.backend == "gxipy":
                label = f"📷 Industrial Camera: {d.name} (Native Daheng USB3)"
            elif d.backend == "directshow":
                label = f"🎥 Video Input: {d.name}"
            elif d.backend == "dataset":
                b_name = os.path.basename(str(d.device_id))
                label = f"📁 Test Recording: Fluid Bed Batch {b_name} ({d.name.split('(')[-1]}"
            else:
                label = f"💻 {d.name}"

            self.cam_combo.addItem(label, d)

        if devices:
            self.cam_combo.setCurrentIndex(0)
            self.statusBar().showMessage(f"Discovered {len(devices)} camera devices / test recordings.")
        else:
            self.statusBar().showMessage("No cameras detected.")

    def toggle_camera_connection(self):
        """Connects or disconnects the selected camera."""
        if self.camera_manager.is_connected:
            self.stop_inspection()
            self.camera_manager.disconnect()
            self.connect_cam_btn.setText("Connect Camera")
            self.connect_cam_btn.setStyleSheet("""
                QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1f3a5f, stop:1 #13243d); color: #00d2ff; border: 1.5px solid #00d2ff; font-weight: 700; border-radius: 6px; padding: 6px 14px; }
                QPushButton:hover { background: #00d2ff; color: #0c1a2e; }
            """)
            self.start_btn.setEnabled(False)
            self.cam_combo.setEnabled(True)
            self.refresh_cam_btn.setEnabled(True)
            self.status_banner.set_status("CAMERA DISCONNECTED", "Select a camera and click 'Connect Camera'", "NORMAL")
            self.statusBar().showMessage("Camera disconnected.")
        else:
            idx = self.cam_combo.currentIndex()
            if idx < 0:
                QMessageBox.warning(self, "Camera", "Please select a camera device from the dropdown.")
                return

            device_info = self.cam_combo.itemData(idx)
            success = self.camera_manager.connect(device_info)
            if success:
                self.connect_cam_btn.setText("Disconnect")
                self.connect_cam_btn.setStyleSheet("""
                    QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #591b1b, stop:1 #381212); color: #ff5252; border: 1.5px solid #ff5252; font-weight: 700; border-radius: 6px; padding: 6px 14px; }
                    QPushButton:hover { background: #ff5252; color: #ffffff; }
                """)
                self.start_btn.setEnabled(True)
                self.cam_combo.setEnabled(False)
                self.refresh_cam_btn.setEnabled(False)
                self.status_banner.set_status("CAMERA CONNECTED", f"Ready on {device_info.name}. Click 'START INSPECTION' to begin.", "NORMAL")
                self.statusBar().showMessage(f"Connected to {device_info.name}")
            else:
                QMessageBox.critical(self, "Connection Failed", f"Could not connect to {device_info.name}.\nPlease ensure the USB cable is securely connected.")

    # ---------------- INSPECTION WORKFLOW ----------------
    def start_inspection(self):
        """Starts real-time AI inspection stream."""
        if not self.camera_manager.is_connected:
            return

        if not self.camera_manager.is_streaming:
            self.camera_manager.start_stream()

        if self.worker is None or not self.worker.isRunning():
            self.worker = InspectionWorker(self.camera_manager, self.pipeline)
            self.worker.frame_ready.connect(self._on_frame_ready)
            self.worker.new_agglomerate_sample.connect(self.gallery_widget.add_sample)
            self.worker.fps_updated.connect(self._on_fps_updated)
            self.worker.alarm_changed.connect(self._on_alarm_changed)
            self.worker.error_occurred.connect(lambda err: self.statusBar().showMessage(f"Notice: {err}"))
            self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_banner.set_status("INSPECTION ACTIVE", "Fluid bed live inspection running.", "NORMAL")
        self.statusBar().showMessage("INSPECTION ACTIVE — Real-time agglomeration and sticking analysis active.")

    def stop_inspection(self):
        """Stops real-time inspection stream."""
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        if self.camera_manager.is_streaming:
            self.camera_manager.stop_stream()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_banner.set_status("INSPECTION PAUSED", "Inspection stopped. Click 'START INSPECTION' to resume.", "NORMAL")
        self.statusBar().showMessage("Inspection Stopped.")

    @Slot(np.ndarray, object)
    def _on_frame_ready(self, annotated: np.ndarray, snapshot: ProcessAnalyticsSnapshot):
        """Receives annotated frame and latest PAT snapshot from worker thread."""
        self._last_snapshot = snapshot
        self.video_widget.update_frame(annotated)
        self._update_operator_displays(snapshot)

    def _update_operator_displays(self, s: ProcessAnalyticsSnapshot):
        """Updates operator gauge and friendly KPI cards with smoothed, non-oscillating metrics."""
        # 1. Circular Process Gauge
        self.circular_gauge.set_value(s.smoothed_agglom_rate)
        trend_str = "Rising ↗" if s.agglom_trend_direction == "RISING" else ("Falling ↘" if s.agglom_trend_direction == "FALLING" else "Stable →")
        self.gauge_sub_lbl.setText(f"{s.smoothed_agglom_rate:.1f}% Agglomeration • {trend_str}")

        # 2. Card 1: Active Flowing Particles
        self.card_active.set_content(
            f"{int(s.smoothed_singles):d} Singles  |  {int(s.smoothed_agglomerates):d} Agglom",
            f"Raw Frame Count: {s.instant_raw_total:d} Total Pellets",
            "#00d2ff"
        )

        # 3. Card 2: Total Pellets Inspected (Unique Cumulative)
        pct_str = f"{s.cumulative_agglom_rate:.1f}%"
        self.card_cumulative.set_content(
            f"{s.cumulative_total_inspected:,d} Pellets",
            f"{s.cumulative_singles:,d} Singles • {s.cumulative_agglomerates:,d} Agglomerates ({pct_str})",
            "#00e676" if s.cumulative_agglom_rate < 15.0 else "#ffaa00"
        )

        # 4. Card 3: Chamber Wall Sticking Risk
        if s.wall_stuck_confirmed_count > 0:
            self.card_stuck.set_content("CRITICAL: Wall Caking!", f"{s.wall_stuck_confirmed_count} adhering clusters detected", "#ff4444")
        elif s.wall_stuck_warning_count > 0:
            self.card_stuck.set_content("WARNING: Wall Adhesion", f"{s.wall_stuck_warning_count} stationary pellet(s)", "#ffaa00")
        else:
            self.card_stuck.set_content("NORMAL: Free Flowing", "Chamber wall clean & free of caking", "#00e676")

        # 5. Card 4: Throughput Rate
        self.card_rate.set_content(
            f"{s.particle_throughput_rate:.1f} Pellets / sec",
            f"Process flow rate (Trend: {s.agglom_trend_direction})",
            "#ffffff"
        )

    def _update_hardware_hud(self):
        cur_dev = self.pipeline.device
        if "cuda" in cur_dev and torch.cuda.is_available():
            idx = int(cur_dev.split(":")[-1]) if ":" in cur_dev else 0
            name = torch.cuda.get_device_name(idx)
            vram = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            self.hw_lbl.setText(f"⚡ GPU: {name} ({vram:.1f} GB)")
            self.hw_lbl.setStyleSheet("color: #00e676; font-size: 10px; font-weight: 700;")
        else:
            self.hw_lbl.setText("💻 CPU Processing (Host System)")
            self.hw_lbl.setStyleSheet("color: #8fa0b5; font-size: 10px;")

    def _on_gpu_device_changed(self, idx: int):
        new_dev = self.gpu_combo.itemData(idx)
        if new_dev and new_dev != self.pipeline.device:
            success = self.pipeline.switch_device(new_dev)
            if success:
                self._update_hardware_hud()
                self.statusBar().showMessage(f"Compute hardware dynamically switched to: {new_dev}")
            else:
                QMessageBox.warning(self, "GPU Switch", f"Could not switch to compute device: {new_dev}")

    @Slot(float, float)
    def _on_fps_updated(self, cam_fps: float, infer_fps: float):
        self.fps_lbl.setText(f"Camera: {cam_fps:.1f} FPS  |  AI: {infer_fps:.1f} FPS")

    @Slot(str, str)
    def _on_alarm_changed(self, level: str, message: str):
        if level == "CRITICAL":
            self.status_banner.set_status("CRITICAL PROCESS ALARM", message, "CRITICAL")
        elif level == "WARNING":
            self.status_banner.set_status("PROCESS WARNING", message, "WARNING")
        else:
            self.status_banner.set_status("OPTIMAL FLUIDIZATION", "Bed fluidization healthy. Low agglomeration.", "NORMAL")

    # ---------------- PARAMETER ADJUSTMENTS ----------------
    def _on_mode_switched(self, idx: int):
        self.settings_stack.setCurrentIndex(idx)

    def _on_sensitivity_preset_changed(self, idx: int):
        presets = [
            (15.0, 30.0),  # Relaxed
            (12.0, 25.0),  # Standard
            (8.0, 18.0)    # Strict
        ]
        warn, crit = presets[min(idx, len(presets) - 1)]
        self.pipeline.analytics_engine.set_thresholds(warn, crit)
        self.circular_gauge.set_thresholds(warn, crit)

    def _on_ema_slider_changed(self, val: int):
        alpha = val / 100.0
        self.pipeline.analytics_engine.set_smoothing_factor(alpha)
        self.ema_lbl.setText(f"{alpha:.2f}")

    def _on_roll_slider_changed(self, val: int):
        self.pipeline.analytics_engine.set_rolling_window(float(val))
        self.roll_lbl.setText(f"{val}.0 Seconds")

    def _on_conf_changed(self, val: int):
        conf = val / 100.0
        self.pipeline.conf_thresh = conf
        self.conf_lbl.setText(f"{conf:.2f}")

    def reset_batch(self):
        self.batch_id = f"BATCH-{int(time.time()) % 100000:05d}"
        self.pipeline.reset_state()
        self.gallery_widget.clear_gallery()
        self.statusBar().showMessage(f"Batch reset. New Batch ID: {self.batch_id}")

    def save_snapshot(self):
        """Saves high-res snapshot of current inspection frame."""
        frame, _, _ = self.camera_manager.get_latest_frame()
        if frame is not None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Inspection Snapshot", f"patvis_snapshot_{int(time.time())}.png", "PNG Images (*.png)"
            )
            if path:
                cv2.imwrite(path, frame)
                QMessageBox.information(self, "Snapshot Saved", f"Successfully saved snapshot to:\n{path}")
        else:
            QMessageBox.warning(self, "Snapshot", "No video frame currently active to capture.")

    def export_batch_report(self):
        """Generates an executive, printable HTML report and opens in browser."""
        if self._last_snapshot is None:
            QMessageBox.warning(self, "Report Export", "Please start inspection first to collect batch data.")
            return

        cam_name = self.cam_combo.currentText()
        samples = self.gallery_widget._samples
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        report_file = os.path.join(out_dir, f"PATVIS_Report_{self.batch_id}.html")

        try:
            BatchReportGenerator.generate_html_report(
                batch_id=self.batch_id,
                snapshot=self._last_snapshot,
                samples=samples,
                camera_name=cam_name,
                output_filepath=report_file
            )
            # Open automatically in default browser
            webbrowser.open(f"file://{os.path.abspath(report_file)}")
            self.statusBar().showMessage(f"Generated Batch Report: {report_file}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not generate report:\n{e}")

    def closeEvent(self, event):
        """Graceful shutdown of inspection worker and camera stream."""
        self.stop_inspection()
        self.camera_manager.disconnect()
        super().closeEvent(event)
