import math
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch

from simulation.pellet_tracker import PelletTracker, KalmanBoxTracker, TrackState
from simulation.reflection_filter import MetalReflectionFilter

class AdhesionState(Enum):
    FREE_FLOWING = 0          # Dynamic movement in fluidization stream (v > 3 px/frame)
    TRANSIENT_INTERACTION = 1 # Momentary deceleration / collision (3 <= age < 7)
    EARLY_STICKING_WARNING = 2# Stationary adherence detected (7 <= age < 15, v < 2.5 px/frame)
    CONFIRMED_WALL_CAKING = 3 # Established wall deposition (age >= 15 or expanding cluster area)


class WallStickingMonitor:
    """
    Industrial Early Agglomeration & Wall Adhesion Monitor for Fluid Bed Processors.
    
    Unified Arbitration Architecture:
      1. Priority 1 (Physical Adhesion):
         - Evaluates stationary persistence near chamber walls and spray nozzles.
         - Any track classified as EARLY_STICKING_WARNING or CONFIRMED_WALL_CAKING is
           EXPLICITLY EXEMPTED from the sequence model. It is never re-evaluated or overwritten.
         - Receives a single authoritative adhesion label ('EARLY STUCK' or 'CAKING').
      2. Priority 2 (Aerodynamic Classification):
         - Non-adhering free-flowing particles are fed to the sequence model (LSTM / Transformer)
           to classify whether they are single pellets, transient optical overlaps, or genuine
           aerodynamic agglomerates.
      3. Metal Reflection Suppression:
         - Specular reflections on metallic walls (X >= 1876) are filtered out beforehand,
           preventing false double-counting and pseudo-dimers.
      4. Single Authoritative Label Guarantee:
         - Every detected particle in the camera view receives EXACTLY ONE unambiguous label.
         - No track is ever evaluated twice or assigned conflicting tags.
    """

    def __init__(
        self,
        tracker: Optional[PelletTracker] = None,
        reflection_filter: Optional[MetalReflectionFilter] = None,
        sequence_model: Optional[Any] = None,
        device: Optional[str] = None,
        min_stuck_frames_warning: int = 7,
        min_stuck_frames_confirmed: int = 15,
        max_stationary_speed: float = 2.8,
        max_stationary_drift: float = 6.5,
        wall_zone_right_x: float = 1750.0,
        wall_zone_left_x: float = 350.0,
        cluster_merge_dist: float = 70.0
    ):
        self.tracker = tracker or PelletTracker(
            high_conf_thresh=0.25,
            low_conf_thresh=0.10,
            iou_thresh=0.15,
            max_lost_frames=10,
            min_hits_to_confirm=2
        )
        self.reflection_filter = reflection_filter or MetalReflectionFilter(wall_x_right=1876.0)
        self.sequence_model = sequence_model
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.sequence_model is not None:
            self.sequence_model.to(self.device)
            self.sequence_model.eval()
            
        self.min_stuck_frames_warning = min_stuck_frames_warning
        self.min_stuck_frames_confirmed = min_stuck_frames_confirmed
        self.max_stationary_speed = max_stationary_speed
        self.max_stationary_drift = max_stationary_drift
        self.wall_zone_right_x = wall_zone_right_x
        self.wall_zone_left_x = wall_zone_left_x
        self.cluster_merge_dist = cluster_merge_dist
        
        # Track adhesion analytics: track_id -> dict of statistics
        self.track_stats: Dict[int, Dict[str, Any]] = {}
        self.frame_index: int = 0
        self.prev_frame_number: Optional[int] = None
        self.alert_history: List[Dict[str, Any]] = []

    def is_in_wall_zone(self, cx: float, cy: float) -> bool:
        """Determines if centroid is in chamber wall or nozzle adhesion zones."""
        return (cx >= self.wall_zone_right_x) or (cx <= self.wall_zone_left_x)

    def process_frame(
        self,
        raw_detections: List[Dict[str, Any]],
        image: Optional[np.ndarray] = None,
        frame_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes a single video frame with strict priority-based arbitration between
        early sticking detection and sequence LSTM / Transformer modeling.
        
        Args:
            raw_detections: Detections from YOLO detector with 'bbox_xyxy', 'confidence', 'class_id'.
            image: Optional BGR image frame for photometric filtering and crop extraction.
            frame_number: Optional integer recording frame index (to detect recording discontinuities).
            
        Returns:
            Dictionary containing active tracks, adhesion states, authoritative labels,
            reflections filtered, and system alert level.
        """
        self.frame_index += 1
        
        # Check for temporal gap between discontinuous bursts (e.g. machine recording paused)
        if frame_number is not None and self.prev_frame_number is not None:
            if frame_number > self.prev_frame_number + 5:
                # Temporal discontinuity: reset tracker to start fresh burst
                self.tracker.reset()
                self.track_stats.clear()
        if frame_number is not None:
            self.prev_frame_number = frame_number

        # Step 1: Suppress Metal Reflections and correct Pseudo-Agglomerates
        clean_dets, suppressed_reflections = self.reflection_filter.filter_detections(raw_detections, image)
        
        # Step 2: Multi-Object ByteTrack association
        active_tracks = self.tracker.update(clean_dets, image=image)
        
        # Step 3: Kinematic Adhesion Analysis & State Determination
        adhesion_states = {}
        adhesion_details = {}
        early_warning_tracks = []
        confirmed_caking_tracks = []
        
        for trk in active_tracks:
            tid = trk.id
            box = trk.get_state_xyxy()
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            w = max(1.0, box[2] - box[0])
            h = max(1.0, box[3] - box[1])
            area = w * h
            
            # Initialize or update track statistics
            if tid not in self.track_stats:
                self.track_stats[tid] = {
                    'first_frame': self.frame_index,
                    'last_frame': self.frame_index,
                    'positions': [(cx, cy)],
                    'areas': [area],
                    'hits': trk.hits,
                    'adhesion_state': AdhesionState.FREE_FLOWING,
                    'warning_frame': None,
                    'confirmed_frame': None,
                    'class_id': trk.history_classes[-1] if trk.history_classes else 0,
                    'authoritative_label': f"P #{tid}",
                    'label_color': (0, 220, 0),
                    'eval_source': "YOLO_INIT",
                    'is_optical_overlap': False
                }
            else:
                stats = self.track_stats[tid]
                stats['last_frame'] = self.frame_index
                stats['positions'].append((cx, cy))
                stats['areas'].append(area)
                stats['hits'] = trk.hits
                stats['class_id'] = trk.history_classes[-1] if trk.history_classes else stats['class_id']
                if len(stats['positions']) > 60:
                    stats['positions'].pop(0)
                    stats['areas'].pop(0)

            stats = self.track_stats[tid]
            recent_pos = stats['positions'][-15:]
            
            # Kinematic Metrics
            if len(recent_pos) >= 2:
                displacements = [
                    math.hypot(recent_pos[i][0] - recent_pos[i-1][0], recent_pos[i][1] - recent_pos[i-1][1])
                    for i in range(1, len(recent_pos))
                ]
                avg_speed = float(np.mean(displacements))
            else:
                vx, vy = trk.get_velocity()
                avg_speed = math.hypot(vx, vy)
                
            mean_cx = float(np.mean([p[0] for p in recent_pos]))
            mean_cy = float(np.mean([p[1] for p in recent_pos]))
            drifts = [math.hypot(p[0] - mean_cx, p[1] - mean_cy) for p in recent_pos]
            max_drift = float(np.max(drifts)) if drifts else 0.0
            
            if len(stats['areas']) >= 6:
                initial_area = np.mean(stats['areas'][:3])
                latest_area = np.mean(stats['areas'][-3:])
                area_growth_ratio = latest_area / max(1.0, initial_area)
            else:
                area_growth_ratio = 1.0

            in_wall = self.is_in_wall_zone(mean_cx, mean_cy)
            is_kinematically_stuck = (avg_speed <= self.max_stationary_speed) and (max_drift <= self.max_stationary_drift)
            is_caking_stuck = is_kinematically_stuck or (in_wall and avg_speed <= self.max_stationary_speed and max_drift <= 12.0)
            
            # Adhesion State Decision
            if stats['hits'] >= self.min_stuck_frames_confirmed and is_caking_stuck:
                state = AdhesionState.CONFIRMED_WALL_CAKING
                if stats['confirmed_frame'] is None:
                    stats['confirmed_frame'] = self.frame_index
                confirmed_caking_tracks.append(tid)
            elif stats['hits'] >= self.min_stuck_frames_warning and (is_kinematically_stuck or (in_wall and avg_speed < 3.2)):
                state = AdhesionState.EARLY_STICKING_WARNING
                if stats['warning_frame'] is None:
                    stats['warning_frame'] = self.frame_index
                early_warning_tracks.append(tid)
            elif stats['hits'] >= 3:
                state = AdhesionState.TRANSIENT_INTERACTION
            else:
                state = AdhesionState.FREE_FLOWING
                
            # Active agglomeration growth upgrade
            if state == AdhesionState.EARLY_STICKING_WARNING and area_growth_ratio > 1.45:
                state = AdhesionState.CONFIRMED_WALL_CAKING
                if tid not in confirmed_caking_tracks:
                    confirmed_caking_tracks.append(tid)
                    if tid in early_warning_tracks:
                        early_warning_tracks.remove(tid)

            stats['adhesion_state'] = state
            adhesion_states[tid] = state

            # =========================================================================
            # MUTUAL EXCLUSION & ARBITRATION PIPELINE
            # =========================================================================
            # Case 1: Track is adhering to the chamber wall / nozzle.
            # PRIORITY 1: Physical adhesion takes precedence over free-flight sequence modeling.
            # Sequence model inference is EXEMPT / BYPASSED to prevent duplicate evaluations.
            if state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.CONFIRMED_WALL_CAKING]:
                stats['eval_source'] = "EARLY_STICKING_MONITOR"
                stats['sequence_eval_skipped'] = True
                stats['is_optical_overlap'] = False
                
                if state == AdhesionState.CONFIRMED_WALL_CAKING:
                    stats['authoritative_label'] = f"CAKING #{tid} (hits={stats['hits']})"
                    stats['label_color'] = (0, 0, 255) # Red
                else:
                    stats['authoritative_label'] = f"EARLY STUCK #{tid} (hits={stats['hits']})"
                    stats['label_color'] = (0, 215, 255) # Yellow

            # Case 2: Track is free-flowing through the fluid bed.
            # PRIORITY 2: Aerodynamic Sequence Model (LSTM / Transformer) to classify
            # free-flying Single vs Agglomerate vs Transient Optical Overlap.
            else:
                stats['sequence_eval_skipped'] = False
                
                if self.sequence_model is not None and len(trk.history_crops) >= 3:
                    stats['eval_source'] = "SEQUENCE_MODEL"
                    
                    # Prepare rolling crops tensor (1, T, 3, 96, 96)
                    crops = trk.history_crops[-8:]
                    crop_tensors = []
                    for c in crops:
                        # Resize to 96x96 if needed and normalize
                        if c.shape[:2] != (96, 96):
                            c_res = cv2.resize(c, (96, 96))
                        else:
                            c_res = c
                        t = torch.from_numpy(c_res).permute(2, 0, 1).float() / 255.0
                        crop_tensors.append(t)
                        
                    crop_seq = torch.stack(crop_tensors).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        pred = self.sequence_model(crop_seq)
                        probs = pred['class_probs'][0].cpu().numpy()
                        ov_prob = float(pred['overlap_prob'][0, 0].cpu().numpy())
                        
                    is_overlap = ov_prob > 0.50
                    seq_class = int(np.argmax(probs))
                    stats['is_optical_overlap'] = is_overlap
                    
                    if is_overlap:
                        stats['authoritative_label'] = f"OVERLAP #{tid}"
                        stats['label_color'] = (255, 220, 100) # Light Blue
                    elif seq_class == 1:
                        stats['authoritative_label'] = f"AGGLOM #{tid}"
                        stats['label_color'] = (255, 165, 0) # Cyan/Orange
                    else:
                        stats['authoritative_label'] = f"P #{tid}"
                        stats['label_color'] = (0, 220, 0) # Green
                else:
                    # Fallback to YOLO single frame detector
                    stats['eval_source'] = "YOLO_SINGLE_FRAME"
                    stats['is_optical_overlap'] = False
                    if stats['class_id'] == 1:
                        stats['authoritative_label'] = f"AGGLOM #{tid}"
                        stats['label_color'] = (255, 165, 0)
                    else:
                        stats['authoritative_label'] = f"P #{tid}"
                        stats['label_color'] = (0, 220, 0)

            adhesion_details[tid] = {
                'track_id': tid,
                'state': state,
                'hits': stats['hits'],
                'center': (cx, cy),
                'mean_anchor': (mean_cx, mean_cy),
                'avg_speed': avg_speed,
                'max_drift': max_drift,
                'area': area,
                'area_growth_ratio': area_growth_ratio,
                'is_in_wall_zone': in_wall,
                'bbox_xyxy': box,
                'class_id': stats['class_id'],
                'eval_source': stats['eval_source'],
                'sequence_eval_skipped': stats.get('sequence_eval_skipped', False),
                'authoritative_label': stats['authoritative_label'],
                'label_color': stats['label_color'],
                'is_optical_overlap': stats.get('is_optical_overlap', False),
                'frames_since_warning': (self.frame_index - stats['warning_frame']) if stats['warning_frame'] else 0
            }

        # Step 4: System Alert Level and Bed Adhesion Index
        num_warning = len(early_warning_tracks)
        num_caking = len(confirmed_caking_tracks)
        
        if num_caking >= 1:
            system_alert_level = "RED"
        elif num_warning >= 1:
            system_alert_level = "YELLOW"
        else:
            system_alert_level = "GREEN"
            
        adhesion_index = min(1.0, (num_warning * 0.25) + (num_caking * 0.50))
        
        result = {
            'frame_index': self.frame_index,
            'active_tracks': active_tracks,
            'adhesion_states': adhesion_states,
            'adhesion_details': adhesion_details,
            'early_warning_tracks': early_warning_tracks,
            'confirmed_caking_tracks': confirmed_caking_tracks,
            'suppressed_reflections': suppressed_reflections,
            'system_alert_level': system_alert_level,
            'bed_adhesion_index': adhesion_index,
            'clean_detections_count': len(clean_dets),
            'suppressed_reflections_count': len(suppressed_reflections)
        }
        
        if system_alert_level in ["YELLOW", "RED"]:
            self.alert_history.append({
                'frame': self.frame_index,
                'level': system_alert_level,
                'warning_count': num_warning,
                'caking_count': num_caking
            })
            
        return result

    def render_hud(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        ground_truth: Optional[List[Dict[str, Any]]] = None,
        show_trajectories: bool = True
    ) -> np.ndarray:
        """
        Renders clean industrial HUD where each particle receives EXACTLY ONE authoritative label.
        Never produces duplicate or conflicting annotations.
        """
        annotated = frame.copy()
        H, W = annotated.shape[:2]
        
        # 1. Chamber Wall Boundary Line
        wall_x = int(self.reflection_filter.wall_x_right)
        cv2.line(annotated, (wall_x, 0), (wall_x, H), (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(annotated, "WALL BOUNDARY", (wall_x - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # 2. Suppressed Metal Reflections (Single Dashed Gray Box)
        for r in result['suppressed_reflections']:
            box = [int(v) for v in r['bbox_xyxy']]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (100, 100, 100), 1)
            cv2.putText(
                annotated, "REFL", (box[0], box[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1
            )

        # 3. Ground Truth Annotations (Purple, rendered underneath detections for clarity)
        if ground_truth:
            for gt in ground_truth:
                gbox = [int(v) for v in gt['box']]
                cv2.rectangle(annotated, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (220, 0, 220), 2)
                cv2.putText(
                    annotated, "GT STICKING", (gbox[0], gbox[3] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 0, 220), 1
                )

        # 4. Active Tracks (Strictly ONE Label and ONE Box Per Track)
        for tid, details in result['adhesion_details'].items():
            box = [int(v) for v in details['bbox_xyxy']]
            state = details['state']
            label = details['authoritative_label']
            color = details['label_color']
            thickness = 2 if state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.CONFIRMED_WALL_CAKING] else 1
            
            # Single bounding box
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, thickness)
            
            # Single label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (box[0], box[1] - lh - 6), (box[0] + lw + 4, box[1]), color, -1)
            text_color = (0, 0, 0) if state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.FREE_FLOWING] else (255, 255, 255)
            cv2.putText(annotated, label, (box[0] + 2, box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color, 1)
            
            # Stationary drift anchor for stuck pellets
            if state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.CONFIRMED_WALL_CAKING]:
                mx, my = int(details['mean_anchor'][0]), int(details['mean_anchor'][1])
                cv2.circle(annotated, (mx, my), 3, color, -1)
                
            # Trajectory path for free-flowing pellets
            if show_trajectories and tid in self.track_stats and state == AdhesionState.FREE_FLOWING:
                pts = self.track_stats[tid]['positions'][-10:]
                for i in range(1, len(pts)):
                    p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                    p2 = (int(pts[i][0]), int(pts[i][1]))
                    cv2.line(annotated, p1, p2, color, 1, cv2.LINE_AA)

        # 5. Top Status Header HUD
        alert_lvl = result['system_alert_level']
        if alert_lvl == "RED":
            hdr_bg = (0, 0, 180)
            status_txt = "CRITICAL: CONFIRMED WALL CAKING DETECTED"
        elif alert_lvl == "YELLOW":
            hdr_bg = (0, 160, 210)
            status_txt = "WARNING: EARLY WALL STICKING NUCLEATION"
        else:
            hdr_bg = (30, 90, 30)
            status_txt = "NORMAL: FREE-FLOWING FLUID BED"
            
        cv2.rectangle(annotated, (0, 0), (W, 46), hdr_bg, -1)
        cv2.putText(
            annotated, f"[EARLY DETECTION MONITOR] {status_txt}",
            (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        
        info_str = (
            f"Frame: {result['frame_index']:04d} | "
            f"Early Warning: {len(result['early_warning_tracks'])} | "
            f"Caking: {len(result['confirmed_caking_tracks'])} | "
            f"Reflections: {result['suppressed_reflections_count']} | "
            f"Adhesion Index: {result['bed_adhesion_index']:.2f}"
        )
        cv2.putText(
            annotated, info_str,
            (W - 740, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1
        )
        
        return annotated

    def reset(self):
        """Resets all tracker and monitor internal states."""
        self.tracker.reset()
        self.track_stats.clear()
        self.frame_index = 0
        self.prev_frame_number = None
        self.alert_history.clear()
