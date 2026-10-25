import os
import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch

from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from ultralytics import YOLO
from simulation.sequence_model import PelletSequenceModel
from simulation.wall_sticking_monitor import WallStickingMonitor, AdhesionState
from simulation.reflection_filter import MetalReflectionFilter
from simulation.pellet_tracker import PelletTracker
from fluid_bed_pat.camera_manager import IndustrialCameraManager
from fluid_bed_pat.analytics_engine import PATAnalyticsEngine, ProcessAnalyticsSnapshot, AgglomerateSample


class HardwareAcceleratorManager:
    """
    Dynamically discovers and configures all available compute hardware:
    NVIDIA GPUs (RTX 3070, 3080, 3090, 4070, 4080, 4090, RTX 50-series, RTX A6000, etc.)
    and CPU fallback.
    Automatically applies TensorFloat-32 and cuDNN benchmark optimizations.
    """
    @staticmethod
    def get_available_devices() -> List[Dict[str, Any]]:
        devices = []
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                prop = torch.cuda.get_device_properties(i)
                mem_gb = prop.total_memory / (1024**3)
                arch = "NVIDIA GPU"
                if prop.major >= 10:
                    arch = "Blackwell / Next-Gen"
                elif prop.major == 9:
                    arch = "Hopper"
                elif prop.major == 8 and prop.minor >= 9:
                    arch = "Ada Lovelace (RTX 40-series)"
                elif prop.major == 8:
                    arch = "Ampere (RTX 30-series)"
                elif prop.major == 7:
                    arch = "Turing (RTX 20-series)"

                label = f"⚡ GPU {i}: {prop.name} ({mem_gb:.1f} GB - {arch})"
                devices.append({
                    'id': i,
                    'device_str': f"cuda:{i}",
                    'name': prop.name,
                    'memory_gb': mem_gb,
                    'compute_capability': f"{prop.major}.{prop.minor}",
                    'architecture': arch,
                    'label': label,
                    'is_gpu': True
                })

        # Always add CPU fallback
        devices.append({
            'id': -1,
            'device_str': "cpu",
            'name': "Host CPU",
            'memory_gb': 0.0,
            'compute_capability': "N/A",
            'architecture': "x86_64",
            'label': "💻 CPU (Host System Fallback)",
            'is_gpu': False
        })
        return devices

    @staticmethod
    def configure_optimal_settings():
        """Enables TensorFloat-32 and cuDNN benchmark for maximum GPU throughput."""
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception as e:
                print(f"[HardwareAcceleratorManager] Notice on CUDA tuning: {e}")

    @staticmethod
    def get_memory_info(device_str: str) -> str:
        if "cuda" in device_str and torch.cuda.is_available():
            try:
                idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
                allocated = torch.cuda.memory_allocated(idx) / (1024**3)
                total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                return f"{allocated:.2f} / {total:.1f} GB"
            except Exception:
                return "Active"
        return "RAM"


class InferencePipeline:
    """
    High-performance real-time Fluid Bed AI Inspection Pipeline.
    Combines:
    1. YOLO11n / custom fine-tuned pellet detector
    2. Spatio-temporal WallStickingMonitor & ReflectionFilter
    3. Multi-object Kalman/ByteTrack tracker
    4. PATAnalyticsEngine (smoothing, trend, morphometry)
    """

    def __init__(
        self,
        yolo_model_path: str = "outputs/models/pellet_detector_best.pt",
        seq_model_path: Optional[str] = "outputs/models/pellet_lstm_best.pt",
        device: Optional[str] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 1024,
        wall_sticking_enabled: bool = True
    ):
        # Auto-configure optimal CUDA settings
        HardwareAcceleratorManager.configure_optimal_settings()

        if device is None:
            # Dynamically select the best available GPU
            devs = HardwareAcceleratorManager.get_available_devices()
            gpu_devs = [d for d in devs if d['is_gpu']]
            if gpu_devs:
                # Pick the GPU with the highest VRAM
                best_gpu = max(gpu_devs, key=lambda d: d['memory_gb'])
                self.device = best_gpu['device_str']
                print(f"[InferencePipeline] Dynamically selected GPU: {best_gpu['name']} ({best_gpu['memory_gb']:.1f} GB VRAM)")
            else:
                self.device = "cpu"
                print("[InferencePipeline] No CUDA GPU found, using CPU.")
        else:
            self.device = device

        self.yolo_model_path = yolo_model_path
        self.seq_model_path = seq_model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.wall_sticking_enabled = wall_sticking_enabled

        # Overlays display toggles
        self.show_boxes: bool = True
        self.show_labels: bool = True
        self.show_trails: bool = True
        self.show_wall_zones: bool = True

        # Load YOLO model
        print(f"[InferencePipeline] Loading YOLO from {self.yolo_model_path} onto {self.device}...")
        self.yolo_model = YOLO(self.yolo_model_path)
        
        # Load Sequence Model if available
        self.seq_model = None
        if seq_model_path and os.path.exists(seq_model_path):
            try:
                ckpt = torch.load(seq_model_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    arch = ckpt.get('arch', 'lstm')
                    d_model = ckpt.get('d_model', 256)
                    num_layers = ckpt.get('num_layers', 4)
                    self.seq_model = PelletSequenceModel(arch=arch, d_model=d_model, num_layers=num_layers)
                    self.seq_model.load_state_dict(ckpt['model_state_dict'])
                    self.seq_model.to(self.device)
                    self.seq_model.eval()
                    print(f"[InferencePipeline] Loaded sequence {arch.upper()} model.")
            except Exception as e:
                print(f"[InferencePipeline] Notice: Sequence model not loaded ({e}).")

        # Initialize Monitor & Filter
        self.refl_filter = MetalReflectionFilter(wall_x_right=1876.0, deep_metal_right_x=1960.0)
        self.tracker = PelletTracker(
            high_conf_thresh=self.conf_thresh,
            low_conf_thresh=0.10,
            iou_thresh=0.20,
            max_lost_frames=8,
            min_hits_to_confirm=2
        )
        self.monitor = WallStickingMonitor(
            tracker=self.tracker,
            reflection_filter=self.refl_filter,
            sequence_model=self.seq_model,
            device=self.device,
            min_stuck_frames_warning=7,
            min_stuck_frames_confirmed=15,
            max_stationary_speed=2.8,
            max_stationary_drift=6.5
        )

        # Initialize PAT Analytics Engine
        self.analytics_engine = PATAnalyticsEngine(
            smoothing_factor_ema=0.15,
            rolling_window_seconds=3.0,
            warning_agglom_threshold=12.0,
            critical_agglom_threshold=25.0
        )

        self.frame_counter: int = 0

    def reset_state(self):
        """Resets tracker and analytics state for new inspection session."""
        self.frame_counter = 0
        self.tracker.reset()
        self.monitor.track_stats.clear()
        self.monitor.frame_index = 0
        self.analytics_engine.reset_session()

    def switch_device(self, new_device_str: str) -> bool:
        """Dynamically switches compute device (e.g. from one GPU to another or CPU) on the fly."""
        try:
            self.device = new_device_str
            print(f"[InferencePipeline] Dynamically switching compute device to {new_device_str}...")
            if hasattr(self, 'yolo_model') and self.yolo_model is not None:
                self.yolo_model.to(self.device)
            if hasattr(self, 'seq_model') and self.seq_model is not None:
                self.seq_model.to(self.device)
            if hasattr(self, 'monitor') and self.monitor is not None:
                self.monitor.device = self.device
            return True
        except Exception as e:
            print(f"[InferencePipeline] Failed to switch device to {new_device_str}: {e}")
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, ProcessAnalyticsSnapshot, List[AgglomerateSample]]:
        """
        Runs complete detection, arbitration, tracking, smoothing, and rendering pipeline.
        Returns: (annotated_image, snapshot, new_samples)
        """
        self.frame_counter += 1
        curr_idx = frame_idx if frame_idx is not None else self.frame_counter
        H, W = frame.shape[:2]

        prev_gallery_len = len(self.analytics_engine.get_recent_samples(100))

        # 1. Run YOLO inference
        results = self.yolo_model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        raw_detections: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)

                for b, conf, c in zip(boxes, confs, clss):
                    c_name = self.yolo_model.names.get(c, "pellet")
                    raw_detections.append({
                        'bbox_xyxy': [float(v) for v in b],
                        'bbox': [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                        'confidence': float(conf),
                        'class_id': int(c),
                        'class_name': c_name
                    })

        # 2. Run Wall Sticking Monitor & Arbitration
        stuck_tracks_list: List[Dict[str, Any]] = []
        active_tracks: List[Any] = []
        clean_detections: List[Dict[str, Any]] = raw_detections

        if self.wall_sticking_enabled:
            monitor_res = self.monitor.process_frame(
                raw_detections=raw_detections,
                image=frame,
                frame_number=curr_idx
            )
            active_tracks = monitor_res.get('active_tracks', [])
            adhesion_details = monitor_res.get('adhesion_details', {})
            
            # Format stuck tracks for analytics
            for tid, det_info in adhesion_details.items():
                st_state = det_info['state']
                if st_state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.CONFIRMED_WALL_CAKING]:
                    stuck_tracks_list.append({
                        'track_id': tid,
                        'state_name': st_state.name,
                        'bbox': det_info['bbox_xyxy'],
                        'center': det_info['center']
                    })

            # Map track IDs onto detections for gallery extraction
            for det in raw_detections:
                dbx = det['bbox']
                dcx = (dbx[0] + dbx[2]) / 2.0
                dcy = (dbx[1] + dbx[3]) / 2.0
                best_tid = -1
                min_d = 40.0
                for trk in active_tracks:
                    tb = trk.get_state_xyxy()
                    tcx = (tb[0] + tb[2]) / 2.0
                    tcy = (tb[1] + tb[3]) / 2.0
                    dist = ((dcx - tcx)**2 + (dcy - tcy)**2)**0.5
                    if dist < min_d:
                        min_d = dist
                        best_tid = trk.id
                det['track_id'] = best_tid

        # 3. Update PAT Analytics Engine
        snapshot = self.analytics_engine.update_frame(
            frame_index=curr_idx,
            frame_image=frame,
            detections=raw_detections,
            active_tracks=active_tracks,
            stuck_tracks=stuck_tracks_list
        )

        # Check if new agglomerate samples were collected
        all_curr_samples = self.analytics_engine.get_recent_samples(100)
        new_samples = all_curr_samples[prev_gallery_len:] if len(all_curr_samples) > prev_gallery_len else []

        # 4. Render Visual Inspection Overlays
        annotated = self._render_overlays(
            frame=frame,
            detections=raw_detections,
            active_tracks=active_tracks,
            snapshot=snapshot
        )

        return annotated, snapshot, new_samples

    def _render_overlays(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        active_tracks: List[Any],
        snapshot: ProcessAnalyticsSnapshot
    ) -> np.ndarray:
        """Renders industrial inspection annotations on frame."""
        canvas = frame.copy()
        H, W = canvas.shape[:2]

        # 1. Chamber Wall & Nozzle Adhesion Boundary lines
        if self.show_wall_zones:
            # Wall boundary on right (e.g. x=1876)
            cv2.line(canvas, (1876, 0), (1876, H), (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas, "CHAMBER WALL ZONE", (1882, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA
            )
            # Wall boundary on left (x=350)
            cv2.line(canvas, (350, 0), (350, H), (140, 140, 140), 1, cv2.LINE_AA)

        # 2. Centroid Motion Velocity Trails
        if self.show_trails and active_tracks:
            for trk in active_tracks:
                if len(trk.history_boxes) < 2:
                    continue
                color = (0, 220, 0)
                if trk.sequence_class_id > 0 and not trk.is_optical_overlap:
                    color = (0, 0, 255)
                elif trk.is_optical_overlap:
                    color = (255, 200, 0)

                points = []
                for box in trk.history_boxes[-12:]:
                    cx = int((box[0] + box[2]) / 2.0)
                    cy = int((box[1] + box[3]) / 2.0)
                    points.append((cx, cy))
                for i in range(1, len(points)):
                    alpha = i / float(len(points))
                    thick = max(1, int(2.0 * alpha))
                    cv2.line(canvas, points[i - 1], points[i], color, thick)

        # 3. Bounding Boxes and Authoritative Labels
        if self.show_boxes:
            for det in detections:
                box = det['bbox']
                x1, y1, x2, y2 = box
                cls_name = det.get('class_name', '')
                cls_id = det.get('class_id', 0)
                conf = det.get('confidence', 0.0)
                tid = det.get('track_id', -1)
                
                is_agglom = ('agglom' in cls_name.lower()) or (cls_id == 1)
                
                # Check sticking status from monitor track stats
                is_stuck = False
                stuck_state = None
                if tid in self.monitor.track_stats:
                    stuck_state = self.monitor.track_stats[tid]['adhesion_state']
                    if stuck_state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.CONFIRMED_WALL_CAKING]:
                        is_stuck = True

                if is_stuck and stuck_state == AdhesionState.CONFIRMED_WALL_CAKING:
                    color = (0, 0, 255)      # Deep Red
                    label = f"ID#{tid} [WALL CAKING]"
                    thick = 3
                elif is_stuck and stuck_state == AdhesionState.EARLY_STICKING_WARNING:
                    color = (0, 140, 255)    # Orange/Amber
                    label = f"ID#{tid} [STICKING WARN]"
                    thick = 2
                elif is_agglom:
                    color = (30, 30, 240)    # Red: Agglomerate
                    label = f"Agglomerate {conf:.2f}" if tid == -1 else f"ID#{tid} [Agglom] {conf:.2f}"
                    thick = 2
                else:
                    color = (0, 220, 0)      # Green: Single pellet
                    label = f"Single {conf:.2f}" if tid == -1 else f"ID#{tid} [Single] {conf:.2f}"
                    thick = 1

                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

                if self.show_labels:
                    font_scale = 0.38
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    badge_y1 = max(0, y1 - lh - 4)
                    badge_y2 = y1
                    cv2.rectangle(canvas, (x1, badge_y1), (x1 + lw + 4, badge_y2), (20, 20, 20), -1)
                    cv2.putText(canvas, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        # 4. Top Real-Time Process HUD Banner
        hud_h = 44
        alert_col = (30, 80, 30)  # Green
        if snapshot.alarm_level == "CRITICAL":
            alert_col = (30, 30, 180)  # Red
        elif snapshot.alarm_level == "WARNING":
            alert_col = (0, 130, 200)  # Amber

        cv2.rectangle(canvas, (0, 0), (W, hud_h), alert_col, -1)
        
        # Left status text
        cv2.putText(
            canvas, f"PAT STATUS: {snapshot.alarm_level} | {snapshot.alarm_message}",
            (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Right metrics text (Smoothed Process Values)
        hud_info = (
            f"Smoothed Agglom: {snapshot.smoothed_agglom_rate:.1f}% | "
            f"Active: {int(snapshot.smoothed_total):d} (S:{int(snapshot.smoothed_singles):d}, A:{int(snapshot.smoothed_agglomerates):d}) | "
            f"Cum. Total: {snapshot.cumulative_total_inspected:d}"
        )
        (tw, _), _ = cv2.getTextSize(hud_info, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(
            canvas, hud_info,
            (max(20, W - tw - 16), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

        return canvas


class InspectionWorker(QThread):
    """
    Dedicated background worker thread for camera grabbing and AI inference.
    Prevents UI freezing and maintains steady real-time throughput.
    """
    frame_ready = Signal(np.ndarray, object)  # (annotated_image, snapshot)
    new_agglomerate_sample = Signal(object)    # (AgglomerateSample)
    fps_updated = Signal(float, float)        # (cam_fps, infer_fps)
    alarm_changed = Signal(str, str)          # (level, message)
    error_occurred = Signal(str)

    def __init__(
        self,
        camera_manager: IndustrialCameraManager,
        pipeline: InferencePipeline
    ):
        super().__init__()
        self.camera_manager = camera_manager
        self.pipeline = pipeline
        self._is_running: bool = False
        self._is_paused: bool = False
        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()
        self._prev_alarm_level: str = "NORMAL"

    def run(self):
        """Main acquisition and inference loop."""
        self._is_running = True
        fps_timer = time.time()
        infer_count = 0
        last_frame_count = -1

        while self._is_running:
            self._mutex.lock()
            while self._is_paused and self._is_running:
                self._pause_cond.wait(self._mutex)
            self._mutex.unlock()

            if not self._is_running:
                break

            # 1. Fetch latest camera frame
            frame, t_stamp, f_count = self.camera_manager.get_latest_frame()
            if frame is None or f_count == last_frame_count:
                # No new frame arrived yet, sleep briefly
                self.msleep(5)
                continue

            last_frame_count = f_count

            # 2. Run inference pipeline
            t_start = time.time()
            try:
                annotated, snapshot, new_samples = self.pipeline.process_frame(frame, frame_idx=f_count)
            except Exception as e:
                self.error_occurred.emit(f"Inference pipeline exception: {e}")
                self.msleep(10)
                continue

            # 3. Emit signals
            self.frame_ready.emit(annotated, snapshot)

            for s in new_samples:
                self.new_agglomerate_sample.emit(s)

            if snapshot.alarm_level != self._prev_alarm_level:
                self._prev_alarm_level = snapshot.alarm_level
                self.alarm_changed.emit(snapshot.alarm_level, snapshot.alarm_message)

            infer_count += 1
            t_now = time.time()
            if t_now - fps_timer >= 1.0:
                infer_fps = infer_count / (t_now - fps_timer)
                cam_fps = self.camera_manager.get_fps()
                self.fps_updated.emit(cam_fps, infer_fps)
                infer_count = 0
                fps_timer = t_now

    def pause(self):
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self._is_paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._is_running = False
        self._is_paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()
        self.wait(2000)
