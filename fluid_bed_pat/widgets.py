import os
import time
import math
import base64
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QPushButton
)
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

from fluid_bed_pat.analytics_engine import ProcessAnalyticsSnapshot, AgglomerateSample


class CircularProcessGauge(QWidget):
    """
    Modern circular arc gauge displaying current Agglomeration % (0 - 100%).
    Color shifts intuitively:
    - 0% - 10% : Emerald Green (Optimal fluidization)
    - 10% - 25%: Industrial Amber (Warning - rising agglomeration)
    - > 25%    : Crimson Red (Critical alarm - severe agglomeration)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value: float = 0.0
        self.warning_thresh: float = 12.0
        self.critical_thresh: float = 25.0
        self.setFixedSize(140, 140)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_value(self, val: float):
        self.value = max(0.0, min(100.0, float(val)))
        self.update()

    def set_thresholds(self, warning: float, critical: float):
        self.warning_thresh = float(warning)
        self.critical_thresh = float(critical)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 12
        rect = QRectF(margin, margin, w - 2 * margin, h - 2 * margin)

        # Background track
        track_pen = QPen(QColor("#202838"), 10)
        track_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(track_pen)
        # 240 degree gauge (from 210 to -30 degrees)
        start_angle = 210 * 16
        span_angle = -240 * 16
        painter.drawArc(rect, start_angle, span_angle)

        # Active progress arc
        pct = self.value / 100.0
        active_span = int(-240 * 16 * pct)

        if self.value >= self.critical_thresh:
            arc_color = QColor("#ff4444")
        elif self.value >= self.warning_thresh:
            arc_color = QColor("#ffaa00")
        else:
            arc_color = QColor("#00d2ff")

        active_pen = QPen(arc_color, 10)
        active_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(active_pen)
        painter.drawArc(rect, start_angle, active_span)

        # Center Percentage Text
        painter.setPen(QColor("#ffffff"))
        val_font = QFont("Segoe UI", 16, QFont.Bold)
        painter.setFont(val_font)
        painter.drawText(QRectF(0, h * 0.28, w, 28), Qt.AlignCenter, f"{self.value:.1f}%")

        # Subtext Label
        painter.setPen(QColor("#8fa0b5"))
        lbl_font = QFont("Segoe UI", 8, QFont.DemiBold)
        painter.setFont(lbl_font)
        painter.drawText(QRectF(0, h * 0.52, w, 18), Qt.AlignCenter, "AGGLOMERATION")

        # Status text below
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        if self.value >= self.critical_thresh:
            painter.setPen(QColor("#ff4444"))
            status_txt = "ALARM"
        elif self.value >= self.warning_thresh:
            painter.setPen(QColor("#ffaa00"))
            status_txt = "WARNING"
        else:
            painter.setPen(QColor("#00e676"))
            status_txt = "OPTIMAL"
        painter.drawText(QRectF(0, h * 0.68, w, 16), Qt.AlignCenter, status_txt)

        painter.end()


class OperatorStatusBanner(QFrame):
    """
    Large illuminated status banner for the factory floor operator.
    Instantly conveys machine state in clear, non-technical language.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self._setup_ui()
        self.set_status("READY", "Camera connected. Ready to start inspection.", "NORMAL")

    def _setup_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(14, 6, 14, 6)
        self.layout.setSpacing(10)

        # Status dot / icon
        self.icon_lbl = QLabel("🟢")
        self.icon_lbl.setStyleSheet("font-size: 16px;")
        self.layout.addWidget(self.icon_lbl)

        # Primary status title
        self.status_title = QLabel("OPTIMAL FLUIDIZATION")
        self.status_title.setStyleSheet("font-size: 13px; font-weight: 800; letter-spacing: 0.5px;")
        self.layout.addWidget(self.status_title)

        self.sep = QLabel("•")
        self.sep.setStyleSheet("color: #556275; font-size: 14px;")
        self.layout.addWidget(self.sep)

        # Plain language explanation
        self.detail_lbl = QLabel("Single pellets flowing freely. No sticking detected.")
        self.detail_lbl.setStyleSheet("font-size: 11px; color: #d1d9e2;")
        self.layout.addWidget(self.detail_lbl)

        self.layout.addStretch()

        # Batch state pill
        self.badge_lbl = QLabel("BATCH RUNNING")
        self.badge_lbl.setStyleSheet("""
            background-color: rgba(0, 210, 255, 0.15);
            color: #00d2ff;
            border: 1px solid #00d2ff;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 10px;
            font-weight: 700;
        """)
        self.layout.addWidget(self.badge_lbl)

    def set_status(self, title: str, details: str, level: str = "NORMAL"):
        self.status_title.setText(title)
        self.detail_lbl.setText(details)

        if level == "CRITICAL":
            self.icon_lbl.setText("🔴")
            self.setStyleSheet("""
                OperatorStatusBanner {
                    background-color: #2b1114;
                    border: 1.5px solid #ff4444;
                    border-radius: 6px;
                }
            """)
            self.status_title.setStyleSheet("font-size: 13px; font-weight: 800; color: #ff5252;")
            self.badge_lbl.setText("ACTION REQUIRED")
            self.badge_lbl.setStyleSheet("background: #ff4444; color: #ffffff; border-radius: 4px; padding: 3px 10px; font-weight: bold;")
        elif level == "WARNING":
            self.icon_lbl.setText("🟡")
            self.setStyleSheet("""
                OperatorStatusBanner {
                    background-color: #261f0d;
                    border: 1.5px solid #ffaa00;
                    border-radius: 6px;
                }
            """)
            self.status_title.setStyleSheet("font-size: 13px; font-weight: 800; color: #ffaa00;")
            self.badge_lbl.setText("ATTENTION")
            self.badge_lbl.setStyleSheet("background: #ffaa00; color: #000000; border-radius: 4px; padding: 3px 10px; font-weight: bold;")
        else:
            self.icon_lbl.setText("🟢")
            self.setStyleSheet("""
                OperatorStatusBanner {
                    background-color: #0f241a;
                    border: 1.5px solid #00e676;
                    border-radius: 6px;
                }
            """)
            self.status_title.setStyleSheet("font-size: 13px; font-weight: 800; color: #00e676;")
            self.badge_lbl.setText("INSPECTION ACTIVE")
            self.badge_lbl.setStyleSheet("background: rgba(0, 230, 118, 0.15); color: #00e676; border: 1px solid #00e676; border-radius: 4px; padding: 3px 10px; font-weight: bold;")


class ModernOperatorCard(QFrame):
    """
    Sleek, tactile card designed for factory operators with icon, title,
    prominent readable value, and supportive context.
    """

    def __init__(self, icon: str, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            ModernOperatorCard {
                background-color: #181d2a;
                border: 1px solid #283347;
                border-radius: 8px;
            }
            ModernOperatorCard:hover {
                border-color: #00d2ff;
                background-color: #1c2232;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header with icon and title
        hdr = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 13px;")
        hdr.addWidget(icon_lbl)

        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet("font-size: 9px; font-weight: 700; color: #8fa0b5; letter-spacing: 0.5px;")
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        layout.addLayout(hdr)

        # Primary value
        self.val_lbl = QLabel("--")
        self.val_lbl.setStyleSheet("font-size: 19px; font-weight: 800; color: #ffffff;")
        layout.addWidget(self.val_lbl)

        # Supportive subtitle
        self.sub_lbl = QLabel("--")
        self.sub_lbl.setStyleSheet("font-size: 10px; color: #6e829e;")
        layout.addWidget(self.sub_lbl)

    def set_content(self, value: str, subtitle: str, value_color: str = "#ffffff"):
        self.val_lbl.setText(value)
        self.val_lbl.setStyleSheet(f"font-size: 19px; font-weight: 800; color: {value_color};")
        self.sub_lbl.setText(subtitle)


class BatchReportGenerator:
    """
    Produces clean, self-contained, printable HTML inspection reports
    suitable for QA compliance, audit trails, and non-technical stakeholders.
    """

    @staticmethod
    def generate_html_report(
        batch_id: str,
        snapshot: ProcessAnalyticsSnapshot,
        samples: List[AgglomerateSample],
        camera_name: str,
        output_filepath: str
    ) -> str:
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.timestamp))
        total_pellets = snapshot.cumulative_total_inspected
        agglom_rate = snapshot.cumulative_agglom_rate
        verdict = "PASS" if snapshot.alarm_level == "NORMAL" else ("CONDITIONAL" if snapshot.alarm_level == "WARNING" else "FAIL")
        verdict_color = "#00e676" if verdict == "PASS" else ("#ffaa00" if verdict == "CONDITIONAL" else "#ff4444")

        # Encode top 12 samples to base64 for pure standalone HTML
        sample_cards_html = ""
        for s in samples[:12]:
            crop = s.crop_image
            if crop is not None and crop.size > 0:
                _, buf = cv2.imencode(".png", crop)
                b64 = base64.b64encode(buf).decode("utf-8")
                stuck_tag = "<span style='color:#ff4444; font-weight:bold;'>[WALL STUCK]</span>" if s.is_wall_stuck else ""
                sample_cards_html += f"""
                <div class="sample-card">
                    <img src="data:image/png;base64,{b64}" alt="Agglomerate" />
                    <div class="sample-meta">
                        <strong>Track ID #{s.track_id}</strong> {stuck_tag}<br>
                        Confidence: {s.confidence*100:.0f}%<br>
                        Diameter: {s.equiv_diameter_px:.1f} px<br>
                        Circularity: {s.circularity:.2f}
                    </div>
                </div>
                """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sensum PATVIS APA — Fluid Bed Inspection Report (Batch {batch_id})</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #f7f9fa;
        color: #2c3e50;
        margin: 0;
        padding: 30px;
    }}
    .report-container {{
        max-width: 950px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        padding: 36px;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #00d2ff;
        padding-bottom: 20px;
        margin-bottom: 25px;
    }}
    .brand-title {{
        font-size: 24px;
        font-weight: 800;
        color: #0c2340;
    }}
    .brand-sub {{
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
    }}
    .verdict-badge {{
        background-color: {verdict_color};
        color: #ffffff;
        padding: 8px 24px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 1px;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }}
    .kpi-box {{
        background: #f1f5f9;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .kpi-box h4 {{
        margin: 0 0 6px 0;
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .kpi-box .val {{
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
    }}
    .section-title {{
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 30px;
        margin-bottom: 12px;
        border-left: 4px solid #00d2ff;
        padding-left: 10px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 25px;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #e2e8f0;
        font-size: 13px;
    }}
    th {{
        background-color: #f8fafc;
        color: #475569;
        font-weight: 600;
    }}
    .gallery-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 15px;
    }}
    .sample-card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
    }}
    .sample-card img {{
        max-width: 100%;
        height: 100px;
        object-fit: contain;
        background: #0f172a;
        border-radius: 4px;
    }}
    .sample-meta {{
        font-size: 10px;
        color: #475569;
        margin-top: 6px;
        line-height: 1.4;
    }}
    .print-btn {{
        background: #00d2ff;
        color: #0c2340;
        border: none;
        padding: 10px 22px;
        font-size: 13px;
        font-weight: 700;
        border-radius: 6px;
        cursor: pointer;
        float: right;
        margin-bottom: 20px;
    }}
    @media print {{
        body {{ padding: 0; background: #fff; }}
        .print-btn {{ display: none; }}
        .report-container {{ box-shadow: none; padding: 0; }}
    }}
</style>
</head>
<body>
<div class="report-container">
    <button class="print-btn" onclick="window.print()">🖨️ Print / Save PDF</button>
    <div class="header">
        <div>
            <div class="brand-title">SENSUM PATVIS APA</div>
            <div class="brand-sub">Fluid Bed In-Line Process Analytical Technology Report</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Batch: <strong>{batch_id}</strong> | Timestamp: {timestamp_str}</div>
        </div>
        <div class="verdict-badge">{verdict}</div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-box">
            <h4>Total Particles</h4>
            <div class="val">{total_pellets:,d}</div>
        </div>
        <div class="kpi-box">
            <h4>Agglomeration Rate</h4>
            <div class="val" style="color: {verdict_color};">{agglom_rate:.1f}%</div>
        </div>
        <div class="kpi-box">
            <h4>Wall Sticking Incidents</h4>
            <div class="val">{snapshot.wall_stuck_confirmed_count + snapshot.wall_stuck_warning_count}</div>
        </div>
        <div class="kpi-box">
            <h4>Throughput Rate</h4>
            <div class="val">{snapshot.particle_throughput_rate:.1f}/s</div>
        </div>
    </div>

    <div class="section-title">Process Inspection Summary</div>
    <table>
        <tr><th>Parameter</th><th>Recorded Value</th><th>Specification / Target</th></tr>
        <tr><td>Inspection Camera</td><td>{camera_name}</td><td>Daheng MER2-301-125U3M (3.1 MP Monochrome)</td></tr>
        <tr><td>Single Pellets Inspected</td><td>{snapshot.cumulative_singles:,d}</td><td>Continuous flow yield</td></tr>
        <tr><td>Confirmed Agglomerates</td><td>{snapshot.cumulative_agglomerates:,d}</td><td>Minimal (< 10%)</td></tr>
        <tr><td>Smoothed Agglomeration Rate</td><td>{snapshot.smoothed_agglom_rate:.1f}%</td><td>Action Limit: < 25.0%</td></tr>
        <tr><td>Wall Adhesion Status</td><td>{snapshot.alarm_message}</td><td>Free-flowing fluidization</td></tr>
        <tr><td>Process Health Verdict</td><td><strong>{verdict}</strong></td><td>Quality Release Standard</td></tr>
    </table>

    <div class="section-title">Representative Captured Agglomerates</div>
    <div class="gallery-grid">
        {sample_cards_html if sample_cards_html else "<p style='color:#64748b;'>No significant agglomerates detected during this inspection period.</p>"}
    </div>

    <div style="margin-top: 40px; font-size: 10px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 14px;">
        Generated automatically by Sensum PATVIS APA Inspection System • Fluid Bed Process Analytical Technology
    </div>
</div>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_filepath
