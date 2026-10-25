import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Deque
import numpy as np
import cv2

@dataclass
class AgglomerateSample:
    """Represents an extracted physical agglomerate sample for the PAT gallery."""
    sample_id: int
    track_id: int
    timestamp: float
    frame_index: int
    crop_image: np.ndarray  # BGR crop
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    area_px: float
    equiv_diameter_px: float
    aspect_ratio: float
    circularity: float
    is_wall_stuck: bool = False
    notes: str = ""


@dataclass
class FrameDetectionRecord:
    timestamp: float
    frame_index: int
    raw_singles: int
    raw_agglomerates: int
    raw_total: int
    raw_agglom_rate: float
    stuck_warning_count: int
    stuck_confirmed_count: int


@dataclass
class ProcessAnalyticsSnapshot:
    """Current consolidated PAT process metrics."""
    timestamp: float
    frame_index: int
    
    # Cumulative Session Totals (Unique tracked particles)
    cumulative_total_inspected: int
    cumulative_singles: int
    cumulative_agglomerates: int
    cumulative_agglom_rate: float  # Percentage: 0.0 - 100.0%
    
    # Instantaneous Raw Counts (Single frame)
    instant_raw_singles: int
    instant_raw_agglomerates: int
    instant_raw_total: int
    instant_raw_agglom_rate: float
    
    # Smoothed Instantaneous Metrics (EMA & Rolling Window - Solves user's oscillation issue!)
    smoothed_singles: float
    smoothed_agglomerates: float
    smoothed_total: float
    smoothed_agglom_rate: float  # Percentage: 0.0 - 100.0%
    
    # Trend Analysis
    agglom_trend_direction: str  # 'RISING', 'STABLE', 'FALLING'
    agglom_rate_of_change: float # % change per second
    particle_throughput_rate: float # particles per second
    
    # Wall Sticking & Chamber Health
    wall_stuck_warning_count: int
    wall_stuck_confirmed_count: int
    alarm_level: str  # 'NORMAL', 'WARNING', 'CRITICAL'
    alarm_message: str


class PATAnalyticsEngine:
    """
    Process Analytical Technology (PAT) Analytics Engine.
    Engineered for fluid bed processors, coating pans, and pelletization chambers.
    
    Solves count oscillation via:
    1. Exponential Moving Average (EMA) filtering
    2. Rolling-window statistical aggregation (1.0s - 5.0s window)
    3. Persistent Kalman trajectory deduplication for true cumulative throughput
    4. Physical morphometry & agglomerate sample gallery management
    """

    def __init__(
        self,
        smoothing_factor_ema: float = 0.12,
        rolling_window_seconds: float = 3.0,
        warning_agglom_threshold: float = 12.0,  # 12% agglomeration warning
        critical_agglom_threshold: float = 25.0, # 25% agglomeration alarm
        max_gallery_samples: int = 150
    ):
        self.ema_alpha = float(smoothing_factor_ema)
        self.rolling_window_sec = float(rolling_window_seconds)
        self.warning_threshold = float(warning_agglom_threshold)
        self.critical_threshold = float(critical_agglom_threshold)
        self.max_gallery_samples = max_gallery_samples
        
        # Cumulative session trackers (Unique track IDs)
        self._tracked_single_ids: set = set()
        self._tracked_agglom_ids: set = set()
        self._tracked_stuck_ids: set = set()
        
        # Smoothed state variables (EMA)
        self._ema_singles: float = 0.0
        self._ema_agglom: float = 0.0
        self._ema_total: float = 0.0
        self._ema_agglom_rate: float = 0.0
        self._ema_initialized: bool = False
        
        # Rolling time window buffer
        self._recent_records: Deque[FrameDetectionRecord] = deque()
        
        # Particle sample gallery
        self._gallery_samples: List[AgglomerateSample] = []
        self._sample_id_counter: int = 0
        self._seen_track_sample_times: Dict[int, float] = {}  # track_id -> last_crop_time
        
        # Session start & throughput calculation
        self._session_start_time = time.time()
        self._total_frames_processed = 0
        self._last_snapshot_time = time.time()

    def reset_session(self):
        """Resets all cumulative metrics and gallery for a new batch."""
        self._tracked_single_ids.clear()
        self._tracked_agglom_ids.clear()
        self._tracked_stuck_ids.clear()
        self._ema_singles = 0.0
        self._ema_agglom = 0.0
        self._ema_total = 0.0
        self._ema_agglom_rate = 0.0
        self._ema_initialized = False
        self._recent_records.clear()
        self._gallery_samples.clear()
        self._sample_id_counter = 0
        self._seen_track_sample_times.clear()
        self._session_start_time = time.time()
        self._total_frames_processed = 0

    def set_smoothing_factor(self, alpha: float):
        """Adjusts the EMA smoothing constant (0.01 - 0.50)."""
        self.ema_alpha = max(0.01, min(0.50, float(alpha)))

    def set_rolling_window(self, window_sec: float):
        """Adjusts rolling window duration in seconds."""
        self.rolling_window_sec = max(0.5, min(15.0, float(window_sec)))

    def set_thresholds(self, warning_pct: float, critical_pct: float):
        """Sets agglomeration alarm threshold percentages."""
        self.warning_threshold = float(warning_pct)
        self.critical_threshold = float(critical_pct)

    def update_frame(
        self,
        frame_index: int,
        frame_image: Optional[np.ndarray],
        detections: List[Dict[str, Any]],
        active_tracks: Optional[List[Any]] = None,
        stuck_tracks: Optional[List[Dict[str, Any]]] = None
    ) -> ProcessAnalyticsSnapshot:
        """
        Ingests frame detections and tracking states, updates EMA, rolling windows,
        registers unique particles, and extracts agglomerate samples.
        """
        t_now = time.time()
        self._total_frames_processed += 1
        
        # 1. Count instantaneous raw detections
        raw_singles = 0
        raw_agglomerates = 0
        
        for det in detections:
            cls_name = det.get('class_name', '')
            cls_id = det.get('class_id', 0)
            if 'agglom' in cls_name.lower() or cls_id == 1:
                raw_agglomerates += 1
            else:
                raw_singles += 1

        raw_total = raw_singles + raw_agglomerates
        raw_agglom_rate = (raw_agglomerates / raw_total * 100.0) if raw_total > 0 else 0.0

        stuck_warning = 0
        stuck_confirmed = 0
        if stuck_tracks:
            for st in stuck_tracks:
                st_state = st.get('state_name', '')
                if 'CAKING' in st_state or 'CONFIRMED' in st_state:
                    stuck_confirmed += 1
                elif 'WARNING' in st_state or 'STICKING' in st_state:
                    stuck_warning += 1

        # 2. Update persistent unique track counts (Kalman Tracker Deduplication)
        if active_tracks:
            for trk in active_tracks:
                t_id = getattr(trk, 'id', None)
                if t_id is None:
                    continue
                
                # Check classification
                seq_cls = getattr(trk, 'sequence_class_id', 0)
                is_overlap = getattr(trk, 'is_optical_overlap', False)
                is_agglom = (seq_cls > 0) and not is_overlap
                
                if is_agglom:
                    self._tracked_agglom_ids.add(t_id)
                    # If previously in singles, migrate to agglomerate
                    self._tracked_single_ids.discard(t_id)
                else:
                    if t_id not in self._tracked_agglom_ids:
                        self._tracked_single_ids.add(t_id)

        # 3. Update Exponential Moving Average (EMA Smoothing)
        if not self._ema_initialized:
            self._ema_singles = float(raw_singles)
            self._ema_agglom = float(raw_agglomerates)
            self._ema_total = float(raw_total)
            self._ema_agglom_rate = float(raw_agglom_rate)
            self._ema_initialized = True
        else:
            a = self.ema_alpha
            self._ema_singles = a * raw_singles + (1.0 - a) * self._ema_singles
            self._ema_agglom = a * raw_agglomerates + (1.0 - a) * self._ema_agglom
            self._ema_total = a * raw_total + (1.0 - a) * self._ema_total
            
            # Smoothed agglomeration percentage
            current_rate = (self._ema_agglom / self._ema_total * 100.0) if self._ema_total > 0.5 else 0.0
            self._ema_agglom_rate = a * current_rate + (1.0 - a) * self._ema_agglom_rate

        # 4. Rolling time-window buffer maintenance
        record = FrameDetectionRecord(
            timestamp=t_now,
            frame_index=frame_index,
            raw_singles=raw_singles,
            raw_agglomerates=raw_agglomerates,
            raw_total=raw_total,
            raw_agglom_rate=raw_agglom_rate,
            stuck_warning_count=stuck_warning,
            stuck_confirmed_count=stuck_confirmed
        )
        self._recent_records.append(record)
        
        # Purge records older than rolling_window_sec
        cutoff_time = t_now - self.rolling_window_sec
        while self._recent_records and self._recent_records[0].timestamp < cutoff_time:
            self._recent_records.popleft()

        # 5. Extract Agglomerate Samples for the Sensum PAT Gallery
        if frame_image is not None and raw_agglomerates > 0:
            self._extract_agglomerate_samples(
                frame_image=frame_image,
                detections=detections,
                frame_index=frame_index,
                timestamp=t_now
            )

        # 6. Trend calculation (Linear regression slope over rolling window)
        agglom_trend_dir = "STABLE"
        agglom_roc = 0.0
        if len(self._recent_records) >= 8:
            times = np.array([r.timestamp for r in self._recent_records])
            rates = np.array([r.raw_agglom_rate for r in self._recent_records])
            dt = times[-1] - times[0]
            if dt > 0.5:
                # Simple linear fit slope (% per second)
                slope, _ = np.polyfit(times - times[0], rates, 1)
                agglom_roc = float(slope)
                if agglom_roc > 1.2:
                    agglom_trend_dir = "RISING"
                elif agglom_roc < -1.2:
                    agglom_trend_dir = "FALLING"

        # 7. Throughput calculation (pellets inspected per second)
        session_duration = max(1.0, t_now - self._session_start_time)
        cum_total = len(self._tracked_single_ids) + len(self._tracked_agglom_ids)
        cum_singles = len(self._tracked_single_ids)
        cum_agglom = len(self._tracked_agglom_ids)
        cum_agglom_rate = (cum_agglom / cum_total * 100.0) if cum_total > 0 else 0.0
        throughput = cum_total / session_duration

        # 8. Alarm Level Determination
        alarm_level = "NORMAL"
        alarm_msg = "Fluidization stable. Low agglomeration."

        if stuck_confirmed > 0:
            alarm_level = "CRITICAL"
            alarm_msg = f"CRITICAL: {stuck_confirmed} wall caking / adhered agglomerates detected!"
        elif self._ema_agglom_rate >= self.critical_threshold:
            alarm_level = "CRITICAL"
            alarm_msg = f"CRITICAL: High agglomeration rate ({self._ema_agglom_rate:.1f}% >= {self.critical_threshold:.1f}%)"
        elif stuck_warning > 0:
            alarm_level = "WARNING"
            alarm_msg = f"WARNING: {stuck_warning} stationary pellet(s) adhering near wall."
        elif self._ema_agglom_rate >= self.warning_threshold:
            alarm_level = "WARNING"
            alarm_msg = f"WARNING: Elevated agglomeration ({self._ema_agglom_rate:.1f}% >= {self.warning_threshold:.1f}%)"

        return ProcessAnalyticsSnapshot(
            timestamp=t_now,
            frame_index=frame_index,
            cumulative_total_inspected=cum_total,
            cumulative_singles=cum_singles,
            cumulative_agglomerates=cum_agglom,
            cumulative_agglom_rate=cum_agglom_rate,
            instant_raw_singles=raw_singles,
            instant_raw_agglomerates=raw_agglomerates,
            instant_raw_total=raw_total,
            instant_raw_agglom_rate=raw_agglom_rate,
            smoothed_singles=self._ema_singles,
            smoothed_agglomerates=self._ema_agglom,
            smoothed_total=self._ema_total,
            smoothed_agglom_rate=self._ema_agglom_rate,
            agglom_trend_direction=agglom_trend_dir,
            agglom_rate_of_change=agglom_roc,
            particle_throughput_rate=throughput,
            wall_stuck_warning_count=stuck_warning,
            wall_stuck_confirmed_count=stuck_confirmed,
            alarm_level=alarm_level,
            alarm_message=alarm_msg
        )

    def _extract_agglomerate_samples(
        self,
        frame_image: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_index: int,
        timestamp: float
    ):
        """Extracts high-resolution crops and morphometry for detected agglomerates."""
        H, W = frame_image.shape[:2]
        
        for det in detections:
            cls_name = det.get('class_name', '')
            cls_id = det.get('class_id', 0)
            if 'agglom' not in cls_name.lower() and cls_id != 1:
                continue

            track_id = det.get('track_id', -1)
            # Throttle same track ID to avoid flooding the gallery with consecutive frames
            if track_id != -1:
                last_time = self._seen_track_sample_times.get(track_id, 0.0)
                if (timestamp - last_time) < 1.0:  # Maximum 1 sample per second per track
                    continue
                self._seen_track_sample_times[track_id] = timestamp

            box = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in box]
            w = x2 - x1
            h = y2 - y1
            if w <= 4 or h <= 4:
                continue

            # Add 25% margin padding for contextual inspection
            pad_x = int(w * 0.25)
            pad_y = int(h * 0.25)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(W, x2 + pad_x)
            crop_y2 = min(H, y2 + pad_y)

            crop = frame_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if crop.size == 0:
                continue

            # Morphometric analysis via contour extraction
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            # Otsu thresholding or adaptive threshold to segment pellet body
            _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(c))
                peri = float(cv2.arcLength(c, True))
            else:
                area = float(w * h)
                peri = float(2 * (w + h))

            if area < 10.0:
                area = float(w * h)
            if peri <= 0.0:
                peri = float(2 * (w + h))

            equiv_diam = 2.0 * math.sqrt(area / math.pi)
            aspect_ratio = float(w) / float(h) if h > 0 else 1.0
            circularity = (4.0 * math.pi * area) / (peri * peri) if peri > 0 else 0.0
            circularity = min(1.0, max(0.0, circularity))

            self._sample_id_counter += 1
            sample = AgglomerateSample(
                sample_id=self._sample_id_counter,
                track_id=track_id,
                timestamp=timestamp,
                frame_index=frame_index,
                crop_image=crop,
                confidence=float(det.get('confidence', 0.85)),
                bbox=[x1, y1, x2, y2],
                area_px=area,
                equiv_diameter_px=equiv_diam,
                aspect_ratio=aspect_ratio,
                circularity=circularity,
                is_wall_stuck=bool(det.get('is_stuck', False)),
                notes=f"Aspect: {aspect_ratio:.2f}, Circ: {circularity:.2f}"
            )

            self._gallery_samples.append(sample)
            if len(self._gallery_samples) > self.max_gallery_samples:
                self._gallery_samples.pop(0)

    def get_recent_samples(self, count: int = 30) -> List[AgglomerateSample]:
        """Returns the most recent agglomerate samples."""
        return self._gallery_samples[-count:]

    def get_time_series_data(self) -> Dict[str, np.ndarray]:
        """Extracts aligned time-series arrays for real-time charting."""
        if not self._recent_records:
            return {
                'timestamps': np.array([]),
                'agglom_rates': np.array([]),
                'singles': np.array([]),
                'agglomerates': np.array([]),
                'totals': np.array([])
            }
        
        t0 = self._recent_records[0].timestamp
        rel_times = np.array([r.timestamp - t0 for r in self._recent_records])
        agglom_rates = np.array([r.raw_agglom_rate for r in self._recent_records])
        singles = np.array([r.raw_singles for r in self._recent_records])
        agglomerates = np.array([r.raw_agglomerates for r in self._recent_records])
        totals = np.array([r.raw_total for r in self._recent_records])

        return {
            'timestamps': rel_times,
            'agglom_rates': agglom_rates,
            'singles': singles,
            'agglomerates': agglomerates,
            'totals': totals
        }
