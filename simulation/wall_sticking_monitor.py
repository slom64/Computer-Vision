import math
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

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
    
    Identifies early adhesion events (single pellets or nascent clusters adhering to chamber
    walls or spray nozzle structures) BEFORE catastrophic fluid-bed caking/collapse occurs.
    
    Key Features:
      - Integrated Metal Reflection Filtering: Suppresses mirror images and splits pseudo-dimers.
      - Temporal Kinematic Analysis: Centroid drift D < 6.0 px, mean velocity v < 2.5 px/frame.
      - Persistence Threshold: Flags EARLY_STICKING_WARNING at 7+ frames (client specification).
      - Growth Rate Monitor: Tracks bounding box area dA/dt to detect active cake accretion.
      - Chamber Wall Zonation: Enforces high sensitivity near right/left metallic walls & nozzles.
      - System Alert Levels: GREEN (Normal), YELLOW (Early Warning), RED (Critical Caking).
    """

    def __init__(
        self,
        tracker: Optional[PelletTracker] = None,
        reflection_filter: Optional[MetalReflectionFilter] = None,
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
        self.alert_history: List[Dict[str, Any]] = []

    def is_in_wall_zone(self, cx: float, cy: float) -> bool:
        """Determines if centroid is in chamber wall or nozzle adhesion zones."""
        return (cx >= self.wall_zone_right_x) or (cx <= self.wall_zone_left_x)

    def process_frame(
        self,
        raw_detections: List[Dict[str, Any]],
        image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes a single video frame through the full detection, reflection suppression,
        tracking, and wall adhesion monitoring pipeline.
        
        Args:
            raw_detections: Detections from YOLO detector with 'bbox_xyxy', 'confidence', 'class_id'.
            image: Optional BGR image frame for photometric filtering and crop extraction.
            
        Returns:
            Dictionary containing:
                'active_tracks': List of active KalmanBoxTrackers
                'adhesion_states': Dict mapping track_id -> AdhesionState
                'adhesion_details': Dict mapping track_id -> detailed metrics
                'early_warning_tracks': List of track_ids in EARLY_STICKING_WARNING
                'confirmed_caking_tracks': List of track_ids in CONFIRMED_WALL_CAKING
                'suppressed_reflections': List of detections filtered out as metal reflections
                'system_alert_level': 'GREEN', 'YELLOW', or 'RED'
                'bed_adhesion_index': Float from 0.0 to 1.0
        """
        self.frame_index += 1
        
        # Step 1: Suppress Metal Reflections and correct Pseudo-Agglomerates
        clean_dets, suppressed_reflections = self.reflection_filter.filter_detections(raw_detections, image)
        
        # Step 2: Multi-Object ByteTrack association
        active_tracks = self.tracker.update(clean_dets, image=image)
        
        # Step 3: Update adhesion analytics for each track
        adhesion_states = {}
        adhesion_details = {}
        early_warning_tracks = []
        confirmed_caking_tracks = []
        
        current_active_ids = set()
        
        for trk in active_tracks:
            tid = trk.id
            current_active_ids.add(tid)
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
                    'class_id': trk.history_classes[-1] if trk.history_classes else 0
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
            
            # Compute kinematic metrics
            # 1. Average step speed over recent frames
            if len(recent_pos) >= 2:
                displacements = [
                    math.hypot(recent_pos[i][0] - recent_pos[i-1][0], recent_pos[i][1] - recent_pos[i-1][1])
                    for i in range(1, len(recent_pos))
                ]
                avg_speed = float(np.mean(displacements))
            else:
                vx, vy = trk.get_velocity()
                avg_speed = math.hypot(vx, vy)
                
            # 2. Maximum drift from centroid anchor
            mean_cx = float(np.mean([p[0] for p in recent_pos]))
            mean_cy = float(np.mean([p[1] for p in recent_pos]))
            drifts = [math.hypot(p[0] - mean_cx, p[1] - mean_cy) for p in recent_pos]
            max_drift = float(np.max(drifts)) if drifts else 0.0
            
            # 3. Area growth rate (detects active agglomerate growth on wall)
            if len(stats['areas']) >= 6:
                initial_area = np.mean(stats['areas'][:3])
                latest_area = np.mean(stats['areas'][-3:])
                area_growth_ratio = latest_area / max(1.0, initial_area)
            else:
                area_growth_ratio = 1.0

            in_wall = self.is_in_wall_zone(mean_cx, mean_cy)
            
            # Determine Adhesion State
            # Condition for Stationary Adhesion:
            # Low speed (avg_speed < max_stationary_speed) AND Low spatial drift (max_drift < max_stationary_drift)
            is_kinematically_stuck = (avg_speed <= self.max_stationary_speed) and (max_drift <= self.max_stationary_drift)
            
            is_caking_stuck = is_kinematically_stuck or (in_wall and avg_speed <= self.max_stationary_speed and max_drift <= 12.0)
            
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
                
            # Upgrade state if cluster is actively growing on the wall
            if state == AdhesionState.EARLY_STICKING_WARNING and area_growth_ratio > 1.45:
                state = AdhesionState.CONFIRMED_WALL_CAKING
                if tid not in confirmed_caking_tracks:
                    confirmed_caking_tracks.append(tid)
                    if tid in early_warning_tracks:
                        early_warning_tracks.remove(tid)

            stats['adhesion_state'] = state
            adhesion_states[tid] = state
            
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
            
        # Bed Adhesion Index: 0.0 (Clean) to 1.0 (Critical)
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
        show_trajectories: bool = True
    ) -> np.ndarray:
        """
        Renders industrial inspection HUD with color-coded alerts and wall boundaries.
        Colors:
          - Green: Free-flowing single/agglomerate pellets
          - Yellow: EARLY STICKING WARNING (>= 7 frames)
          - Red / Amber: CONFIRMED WALL CAKING (>= 15 frames)
          - Gray dashed: Suppressed Metal Reflection
        """
        annotated = frame.copy()
        H, W = annotated.shape[:2]
        
        # 1. Draw Chamber Wall Boundary
        wall_x = int(self.reflection_filter.wall_x_right)
        cv2.line(annotated, (wall_x, 0), (wall_x, H), (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(annotated, "WALL BOUNDARY", (wall_x - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # 2. Draw Suppressed Metal Reflections in dashed Gray
        for r in result['suppressed_reflections']:
            box = [int(v) for v in r['bbox_xyxy']]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (100, 100, 100), 1)
            cv2.putText(
                annotated, "REFL", (box[0], box[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1
            )

        # 3. Draw Active Tracks with Adhesion Color Coding
        for tid, details in result['adhesion_details'].items():
            box = [int(v) for v in details['bbox_xyxy']]
            state = details['state']
            hits = details['hits']
            
            if state == AdhesionState.CONFIRMED_WALL_CAKING:
                color = (0, 0, 255) # Bright Red
                label = f"CAKING #{tid} (hits={hits})"
                thickness = 2
            elif state == AdhesionState.EARLY_STICKING_WARNING:
                color = (0, 215, 255) # Bright Yellow / Gold (BGR)
                label = f"EARLY STUCK #{tid} (hits={hits})"
                thickness = 2
            elif details['class_id'] == 1:
                color = (255, 165, 0) # Cyan/Orange for agglomerate
                label = f"AGGLOM #{tid}"
                thickness = 1
            else:
                color = (0, 220, 0) # Green for free single
                label = f"P #{tid}"
                thickness = 1
                
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, thickness)
            
            # Label background banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (box[0], box[1] - lh - 6), (box[0] + lw + 4, box[1]), color, -1)
            text_color = (0, 0, 0) if state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.FREE_FLOWING] else (255, 255, 255)
            cv2.putText(annotated, label, (box[0] + 2, box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color, 1)
            
            # Draw stationary drift anchor
            if state in [AdhesionState.EARLY_STICKING_WARNING, AdhesionState.CONFIRMED_WALL_CAKING]:
                mx, my = int(details['mean_anchor'][0]), int(details['mean_anchor'][1])
                cv2.circle(annotated, (mx, my), 3, color, -1)
                
            # Draw trajectory path
            if show_trajectories and tid in self.track_stats:
                pts = self.track_stats[tid]['positions'][-12:]
                for i in range(1, len(pts)):
                    p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                    p2 = (int(pts[i][0]), int(pts[i][1]))
                    cv2.line(annotated, p1, p2, color, 1, cv2.LINE_AA)

        # 4. Top Status Header HUD
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
            (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2
        )
        
        info_str = (
            f"Frame: {result['frame_index']:04d} | "
            f"Early Warning: {len(result['early_warning_tracks'])} | "
            f"Caking: {len(result['confirmed_caking_tracks'])} | "
            f"Reflections Filtered: {result['suppressed_reflections_count']} | "
            f"Adhesion Index: {result['bed_adhesion_index']:.2f}"
        )
        cv2.putText(
            annotated, info_str,
            (W - 750, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
        )
        
        return annotated

    def reset(self):
        """Resets all tracker and monitor internal states."""
        self.tracker.reset()
        self.track_stats.clear()
        self.frame_index = 0
        self.alert_history.clear()
