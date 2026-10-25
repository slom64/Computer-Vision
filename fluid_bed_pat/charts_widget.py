import time
from typing import Dict, Any, Optional, List
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QComboBox, QPushButton
)
from PySide6.QtCore import Qt, QTimer

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from fluid_bed_pat.analytics_engine import PATAnalyticsEngine, ProcessAnalyticsSnapshot


class PATTrendChartsWidget(QWidget):
    """
    Industrial Real-Time Process Analytical Technology (PAT) Trend Charts.
    Emulates Sensum PATVIS APA process control interface:
    - Tab 1: Real-time Agglomeration % Trend with Warning & Critical Limits
    - Tab 2: Particle Count Dynamics (Singles, Agglomerates, Total)
    - Tab 3: Particle Size / Equivalent Diameter Distribution Histogram (D10, D50, D90)
    """

    def __init__(self, analytics_engine: PATAnalyticsEngine, parent=None):
        super().__init__(parent)
        self.analytics_engine = analytics_engine

        self.history_window_sec: float = 60.0  # 60s scrolling window
        self._setup_ui()

        # Update timer for smooth, throttled chart rendering (4 Hz is standard for industrial charts)
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(250)
        self._update_timer.timeout.connect(self.refresh_charts)
        self._update_timer.start()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header bar with window controls
        hdr_layout = QHBoxLayout()
        title = QLabel("PROCESS ANALYTICS & PAT TRENDS")
        title.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; letter-spacing: 0.5px;")
        hdr_layout.addWidget(title)
        hdr_layout.addStretch()

        win_lbl = QLabel("History:")
        win_lbl.setStyleSheet("color: #8fa0b5; font-size: 11px;")
        hdr_layout.addWidget(win_lbl)

        self.win_combo = QComboBox()
        self.win_combo.addItems(["30 Seconds", "60 Seconds", "3 Minutes", "5 Minutes"])
        self.win_combo.setCurrentIndex(1)
        self.win_combo.currentIndexChanged.connect(self._on_window_changed)
        self.win_combo.setStyleSheet("""
            QComboBox {
                background-color: #1f2533;
                color: #e0e6ed;
                border: 1px solid #333d52;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
        """)
        hdr_layout.addWidget(self.win_combo)

        layout.addLayout(hdr_layout)

        # Tabs for different charts
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #283042;
                background: #151821;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #1b202c;
                color: #8fa0b5;
                padding: 5px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 11px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: #252d3d;
                color: #00d2ff;
                border-bottom: 2px solid #00d2ff;
            }
            QTabBar::tab:hover {
                color: #ffffff;
            }
        """)

        # 1. Agglomeration Trend Tab
        self.fig_agglom = Figure(figsize=(5, 2.5), facecolor="#151821")
        self.canvas_agglom = FigureCanvasQTAgg(self.fig_agglom)
        self.ax_agglom = self.fig_agglom.add_subplot(111, facecolor="#151821")
        self._init_agglom_axis()
        self.tabs.addTab(self.canvas_agglom, "Agglomeration Trend (%)")

        # 2. Particle Dynamics Tab
        self.fig_counts = Figure(figsize=(5, 2.5), facecolor="#151821")
        self.canvas_counts = FigureCanvasQTAgg(self.fig_counts)
        self.ax_counts = self.fig_counts.add_subplot(111, facecolor="#151821")
        self._init_counts_axis()
        self.tabs.addTab(self.canvas_counts, "Particle Flow Dynamics")

        # 3. Size / Equivalent Diameter Distribution Tab
        self.fig_dist = Figure(figsize=(5, 2.5), facecolor="#151821")
        self.canvas_dist = FigureCanvasQTAgg(self.fig_dist)
        self.ax_dist = self.fig_dist.add_subplot(111, facecolor="#151821")
        self._init_dist_axis()
        self.tabs.addTab(self.canvas_dist, "Size Distribution (Deq)")

        layout.addWidget(self.tabs)

    def _init_agglom_axis(self):
        ax = self.ax_agglom
        ax.set_title("Agglomeration Rate (%) vs Elapsed Time", color="#e0e6ed", fontsize=9, pad=6)
        ax.set_ylabel("Agglomeration Rate (%)", color="#8fa0b5", fontsize=8)
        ax.set_xlabel("Time (s)", color="#8fa0b5", fontsize=8)
        ax.tick_params(colors="#8fa0b5", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.25, color="#404c63")
        for spine in ax.spines.values():
            spine.set_color("#283042")
        self.fig_agglom.tight_layout()

    def _init_counts_axis(self):
        ax = self.ax_counts
        ax.set_title("Active Flow Count: Singles vs Agglomerates", color="#e0e6ed", fontsize=9, pad=6)
        ax.set_ylabel("Particle Count", color="#8fa0b5", fontsize=8)
        ax.set_xlabel("Time (s)", color="#8fa0b5", fontsize=8)
        ax.tick_params(colors="#8fa0b5", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.25, color="#404c63")
        for spine in ax.spines.values():
            spine.set_color("#283042")
        self.fig_counts.tight_layout()

    def _init_dist_axis(self):
        ax = self.ax_dist
        ax.set_title("Agglomerate Size Distribution (Equivalent Diameter Deq)", color="#e0e6ed", fontsize=9, pad=6)
        ax.set_ylabel("Frequency", color="#8fa0b5", fontsize=8)
        ax.set_xlabel("Diameter Deq (pixels)", color="#8fa0b5", fontsize=8)
        ax.tick_params(colors="#8fa0b5", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.25, color="#404c63")
        for spine in ax.spines.values():
            spine.set_color("#283042")
        self.fig_dist.tight_layout()

    def _on_window_changed(self, idx: int):
        windows = [30.0, 60.0, 180.0, 300.0]
        self.history_window_sec = windows[min(idx, len(windows) - 1)]

    def refresh_charts(self):
        """Redraws the active tab chart with latest analytics data."""
        curr_tab = self.tabs.currentIndex()
        if curr_tab == 0:
            self._render_agglom_chart()
        elif curr_tab == 1:
            self._render_counts_chart()
        elif curr_tab == 2:
            self._render_dist_chart()

    def _render_agglom_chart(self):
        data = self.analytics_engine.get_time_series_data()
        t = data['timestamps']
        rates = data['agglom_rates']

        self.ax_agglom.cla()
        self._init_agglom_axis()

        warn_lim = self.analytics_engine.warning_threshold
        crit_lim = self.analytics_engine.critical_threshold

        # Plot threshold limit lines (Sensum PAT style)
        self.ax_agglom.axhline(warn_lim, color="#ffaa00", linestyle="--", linewidth=1.2, label=f"Warning ({warn_lim:.0f}%)", alpha=0.85)
        self.ax_agglom.axhline(crit_lim, color="#ff4444", linestyle="--", linewidth=1.4, label=f"Critical Alarm ({crit_lim:.0f}%)", alpha=0.85)

        if len(t) > 1:
            # Filter to history window
            t_max = t[-1]
            mask = t >= (t_max - self.history_window_sec)
            t_win = t[mask] - t_max  # Relative to now: -60s to 0s
            rates_win = rates[mask]

            # Raw instantaneous points
            self.ax_agglom.plot(t_win, rates_win, color="#00d2ff", alpha=0.35, linewidth=1, label="Raw Instantaneous")
            
            # EMA smoothed curve
            if len(rates_win) >= 4:
                # Rolling smoothed trendline
                kernel = np.ones(5) / 5.0
                smoothed = np.convolve(rates_win, kernel, mode='same')
                self.ax_agglom.plot(t_win, smoothed, color="#00d2ff", linewidth=2.2, label="Smoothed Trend (PAT)")
            else:
                self.ax_agglom.plot(t_win, rates_win, color="#00d2ff", linewidth=2.0, label="Smoothed Trend (PAT)")

            y_max = max(crit_lim * 1.3, float(np.max(rates_win) * 1.15 if len(rates_win) > 0 else 30.0))
            self.ax_agglom.set_ylim(0, y_max)
            self.ax_agglom.set_xlim(-self.history_window_sec, 0)
        else:
            self.ax_agglom.set_ylim(0, crit_lim * 1.3)
            self.ax_agglom.set_xlim(-self.history_window_sec, 0)

        self.ax_agglom.legend(loc="upper left", facecolor="#1b202c", edgecolor="#283042", fontsize=7, labelcolor="#e0e6ed")
        self.fig_agglom.tight_layout()
        self.canvas_agglom.draw_idle()

    def _render_counts_chart(self):
        data = self.analytics_engine.get_time_series_data()
        t = data['timestamps']
        singles = data['singles']
        aggloms = data['agglomerates']
        totals = data['totals']

        self.ax_counts.cla()
        self._init_counts_axis()

        if len(t) > 1:
            t_max = t[-1]
            mask = t >= (t_max - self.history_window_sec)
            t_win = t[mask] - t_max

            self.ax_counts.plot(t_win, singles[mask], color="#00e676", linewidth=1.8, label="Singles", alpha=0.9)
            self.ax_counts.plot(t_win, aggloms[mask], color="#ff5252", linewidth=2.0, label="Agglomerates", alpha=0.95)
            self.ax_counts.plot(t_win, totals[mask], color="#ffffff", linestyle=":", linewidth=1.2, label="Total Flow", alpha=0.7)

            y_max = max(20.0, float(np.max(totals[mask]) * 1.2 if np.any(mask) else 50.0))
            self.ax_counts.set_ylim(0, y_max)
            self.ax_counts.set_xlim(-self.history_window_sec, 0)
        else:
            self.ax_counts.set_ylim(0, 100)
            self.ax_counts.set_xlim(-self.history_window_sec, 0)

        self.ax_counts.legend(loc="upper left", facecolor="#1b202c", edgecolor="#283042", fontsize=7, labelcolor="#e0e6ed")
        self.fig_counts.tight_layout()
        self.canvas_counts.draw_idle()

    def _render_dist_chart(self):
        samples = self.analytics_engine.get_recent_samples(100)
        self.ax_dist.cla()
        self._init_dist_axis()

        if samples:
            diams = [s.equiv_diameter_px for s in samples]
            counts, bins, _ = self.ax_dist.hist(diams, bins=12, color="#00d2ff", edgecolor="#151821", alpha=0.75, rwidth=0.85)

            # D10, D50 (median), D90 percentiles
            d10 = float(np.percentile(diams, 10))
            d50 = float(np.percentile(diams, 50))
            d90 = float(np.percentile(diams, 90))

            self.ax_dist.axvline(d10, color="#8fa0b5", linestyle=":", linewidth=1.2, label=f"D10: {d10:.1f}px")
            self.ax_dist.axvline(d50, color="#ffaa00", linestyle="-", linewidth=1.6, label=f"D50 (Median): {d50:.1f}px")
            self.ax_dist.axvline(d90, color="#ff5252", linestyle=":", linewidth=1.2, label=f"D90: {d90:.1f}px")

            self.ax_dist.legend(loc="upper right", facecolor="#1b202c", edgecolor="#283042", fontsize=7, labelcolor="#e0e6ed")
        else:
            self.ax_dist.text(0.5, 0.5, "Awaiting Agglomerate Samples...", color="#667891", ha="center", va="center", transform=self.ax_dist.transAxes, fontsize=10)

        self.fig_dist.tight_layout()
        self.canvas_dist.draw_idle()
