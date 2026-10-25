import os
import time
from typing import List, Optional

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame,
    QPushButton, QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QIcon

from fluid_bed_pat.analytics_engine import AgglomerateSample


class ParticleInspectorDialog(QDialog):
    """
    Sensum PATVIS APA-style High-Resolution Particle Morphometry Inspector.
    Displays zoomed crop, contour boundary, and geometric shape descriptors.
    """

    def __init__(self, sample: AgglomerateSample, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.setWindowTitle(f"Particle Morphometry Inspector — Track ID #{sample.track_id} (Sample #{sample.sample_id})")
        self.resize(750, 480)
        self.setStyleSheet("background-color: #161922; color: #e0e6ed; font-family: Segoe UI, sans-serif;")
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Left Column: Zoomed Particle Image
        left_box = QVBoxLayout()
        img_title = QLabel("HIGH-RESOLUTION OPTICAL CROP")
        img_title.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; letter-spacing: 0.5px;")
        left_box.addWidget(img_title)

        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet("""
            QLabel {
                background-color: #0e1117;
                border: 2px solid #283042;
                border-radius: 6px;
                min-width: 340px;
                min-height: 340px;
            }
        """)
        
        # Render zoomed crop
        crop = self.sample.crop_image
        if crop is not None and crop.size > 0:
            h, w = crop.shape[:2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            # Scale smoothly to fit viewport
            scaled = pix.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_lbl.setPixmap(scaled)

        left_box.addWidget(self.img_lbl)

        btn_box = QHBoxLayout()
        export_btn = QPushButton("Save PNG Crop")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #232a3b;
                color: #00d2ff;
                border: 1px solid #00d2ff;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #00d2ff;
                color: #0e1117;
            }
        """)
        export_btn.clicked.connect(self._save_crop)
        btn_box.addWidget(export_btn)
        left_box.addLayout(btn_box)

        layout.addLayout(left_box, stretch=1)

        # Right Column: Morphometric & Process Descriptors Table
        right_box = QVBoxLayout()
        desc_title = QLabel("PARTICLE MORPHOMETRIC DESCRIPTORS")
        desc_title.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; letter-spacing: 0.5px;")
        right_box.addWidget(desc_title)

        table = QTableWidget(9, 2)
        table.setHorizontalHeaderLabels(["Descriptor", "Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1e29;
                border: 1px solid #283042;
                border-radius: 6px;
                gridline-color: #252c3c;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #232a3b;
                color: #8fa0b5;
                font-weight: bold;
                border: none;
                padding: 6px;
            }
        """)

        s = self.sample
        metrics = [
            ("Track ID", f"#{s.track_id}"),
            ("Sample ID", f"#{s.sample_id}"),
            ("Classification", "Confirmed Agglomerate" if not s.is_wall_stuck else "Wall-Stuck Agglomerate"),
            ("Confidence", f"{s.confidence * 100:.1f}%"),
            ("Equivalent Diameter (Deq)", f"{s.equiv_diameter_px:.1f} px"),
            ("Projected Area", f"{s.area_px:.0f} px²"),
            ("Aspect Ratio (W/H)", f"{s.aspect_ratio:.2f}"),
            ("Circularity Index", f"{s.circularity:.2f} (1.0 = sphere)"),
            ("Chamber Wall Adhesion", "YES (Wall Caking Risk)" if s.is_wall_stuck else "No (Free Flowing)")
        ]

        for row, (desc, val) in enumerate(metrics):
            item_d = QTableWidgetItem(desc)
            item_d.setForeground(QColor("#8fa0b5"))
            item_d.setFlags(item_d.flags() & ~Qt.ItemIsEditable)
            item_v = QTableWidgetItem(val)
            item_v.setForeground(QColor("#00e676" if "YES" in val or "Agglom" in val else "#ffffff"))
            item_v.setFlags(item_v.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, item_d)
            table.setItem(row, 1, item_v)

        right_box.addWidget(table)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b3345;
                color: #e0e6ed;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #3b465e; }
        """)
        close_btn.clicked.connect(self.accept)
        right_box.addWidget(close_btn, alignment=Qt.AlignRight)

        layout.addLayout(right_box, stretch=1)

    def _save_crop(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Agglomerate Crop", f"agglomerate_track_{self.sample.track_id}.png", "PNG Images (*.png)"
        )
        if path:
            cv2.imwrite(path, self.sample.crop_image)
            QMessageBox.information(self, "Export Complete", f"Saved sample image to:\n{path}")


class AgglomerateCardWidget(QFrame):
    """Individual clickable thumbnail card for the horizontal gallery filmstrip."""
    clicked = Signal(object)

    def __init__(self, sample: AgglomerateSample, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.setFixedSize(130, 150)
        self.setCursor(Qt.PointingHandCursor)
        self._setup_ui()

    def _setup_ui(self):
        border_col = "#ff4444" if self.sample.is_wall_stuck else "#ff5252"
        self.setStyleSheet(f"""
            AgglomerateCardWidget {{
                background-color: #1c212d;
                border: 1px solid #2e374a;
                border-radius: 6px;
            }}
            AgglomerateCardWidget:hover {{
                border: 1.5px solid #00d2ff;
                background-color: #222938;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Image thumbnail
        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet("background: #0f1218; border-radius: 4px;")
        self.img_lbl.setFixedHeight(84)

        crop = self.sample.crop_image
        if crop is not None and crop.size > 0:
            h, w = crop.shape[:2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.img_lbl.setPixmap(pix.scaled(118, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        layout.addWidget(self.img_lbl)

        # Metadata labels
        badge = QLabel(f"ID #{self.sample.track_id}  |  {self.sample.confidence*100:.0f}%")
        badge.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 10px;")
        layout.addWidget(badge)

        info = QLabel(f"Deq: {self.sample.equiv_diameter_px:.0f}px  Circ: {self.sample.circularity:.2f}")
        info.setStyleSheet("color: #8fa0b5; font-size: 9px;")
        layout.addWidget(info)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.sample)
        super().mousePressEvent(event)


class AgglomerateGalleryWidget(QWidget):
    """
    Sensum PATVIS APA-style Live Defect / Agglomerate Sample Gallery.
    Displays a horizontal filmstrip of real-time captured agglomerate particles.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_frozen: bool = False
        self._samples: List[AgglomerateSample] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header controls
        hdr = QHBoxLayout()
        title = QLabel("LIVE AGGLOMERATION SAMPLE GALLERY (SENSUM DEFECT STREAM)")
        title.setStyleSheet("font-weight: bold; color: #ff5252; font-size: 11px; letter-spacing: 0.5px;")
        hdr.addWidget(title)

        self.count_lbl = QLabel("(0 Samples Captured)")
        self.count_lbl.setStyleSheet("color: #8fa0b5; font-size: 11px;")
        hdr.addWidget(self.count_lbl)

        hdr.addStretch()

        self.freeze_btn = QPushButton("Pause Stream")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setStyleSheet("""
            QPushButton {
                background: #1f2533;
                color: #e0e6ed;
                border: 1px solid #333d52;
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 11px;
            }
            QPushButton:checked {
                background: #ffaa00;
                color: #0e1117;
                font-weight: bold;
            }
        """)
        self.freeze_btn.toggled.connect(self._toggle_freeze)
        hdr.addWidget(self.freeze_btn)

        export_all_btn = QPushButton("Export Batch Samples")
        export_all_btn.setStyleSheet("""
            QPushButton {
                background: #1f2533;
                color: #00d2ff;
                border: 1px solid #333d52;
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background: #283042; }
        """)
        export_all_btn.clicked.connect(self._export_all_samples)
        hdr.addWidget(export_all_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #1f2533;
                color: #8fa0b5;
                border: 1px solid #333d52;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover { background: #ff4444; color: white; }
        """)
        clear_btn.clicked.connect(self.clear_gallery)
        hdr.addWidget(clear_btn)

        layout.addLayout(hdr)

        # Horizontal Scroll Area for filmstrip
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(175)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("""
            QScrollArea {
                background-color: #12151c;
                border: 1px solid #252c3c;
                border-radius: 6px;
            }
            QScrollBar:horizontal {
                height: 8px;
                background: #151821;
            }
            QScrollBar::handle:horizontal {
                background: #2e374a;
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #00d2ff;
            }
        """)

        self.strip_widget = QWidget()
        self.strip_widget.setStyleSheet("background-color: transparent;")
        self.strip_layout = QHBoxLayout(self.strip_widget)
        self.strip_layout.setContentsMargins(6, 6, 6, 6)
        self.strip_layout.setSpacing(8)
        self.strip_layout.addStretch()

        self.scroll.setWidget(self.strip_widget)
        layout.addWidget(self.scroll)

    def add_sample(self, sample: AgglomerateSample):
        """Appends new agglomerate sample to the filmstrip."""
        if self.is_frozen:
            return

        self._samples.append(sample)
        card = AgglomerateCardWidget(sample)
        card.clicked.connect(self._open_inspector)

        # Insert at the beginning (newest first on left, Sensum style)
        self.strip_layout.insertWidget(0, card)
        self.count_lbl.setText(f"({len(self._samples)} Samples Captured)")

        # Cap widgets at 80 for GUI performance
        if self.strip_layout.count() > 80:
            item = self.strip_layout.takeAt(self.strip_layout.count() - 2)
            if item and item.widget():
                item.widget().deleteLater()

    def _toggle_freeze(self, checked: bool):
        self.is_frozen = checked
        self.freeze_btn.setText("Resume Stream" if checked else "Pause Stream")

    def _open_inspector(self, sample: AgglomerateSample):
        dlg = ParticleInspectorDialog(sample, self)
        dlg.exec()

    def clear_gallery(self):
        self._samples.clear()
        while self.strip_layout.count() > 1:
            item = self.strip_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self.count_lbl.setText("(0 Samples Captured)")

    def _export_all_samples(self):
        if not self._samples:
            QMessageBox.information(self, "Export", "No agglomerate samples available to export.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not folder:
            return

        export_dir = os.path.join(folder, f"PATVIS_Defects_{int(time.time())}")
        os.makedirs(export_dir, exist_ok=True)
        img_dir = os.path.join(export_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        csv_path = os.path.join(export_dir, "agglomerate_summary.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Sample_ID,Track_ID,Confidence,Area_px2,Equiv_Diameter_px,Aspect_Ratio,Circularity,Is_Wall_Stuck,Filename\n")
            for s in self._samples:
                fname = f"sample_{s.sample_id:04d}_track_{s.track_id}.png"
                img_path = os.path.join(img_dir, fname)
                cv2.imwrite(img_path, s.crop_image)
                f.write(f"{s.sample_id},{s.track_id},{s.confidence:.4f},{s.area_px:.1f},{s.equiv_diameter_px:.2f},{s.aspect_ratio:.2f},{s.circularity:.3f},{s.is_wall_stuck},{fname}\n")

        QMessageBox.information(self, "Export Successful", f"Exported {len(self._samples)} samples and CSV report to:\n{export_dir}")
