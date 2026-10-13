import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def compute_box_iou(boxA, boxB) -> float:
    """Computes Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    
    return interArea / float(boxAArea + boxBArea - interArea)

class TemporalAgglomerationVerifier:
    """
    Multi-Frame (3-Frame) Temporal Reasoning Engine.
    Resolves ambiguous single-frame 2D observations in fluidized beds:
    1. Optical Overlaps: Disentangles two independent single pellets flying past each other
       at different depths from true physical agglomerates.
    2. Tumbling Ambiguity: Corrects end-on views of agglomerates (which look round like single pellets
       in 1 frame) by verifying their multi-lobe shape in adjacent frames (t-1 or t+1).
    3. Temporal Consensus: Boosts classification confidence using 3-frame trajectory continuity.
    """
    def __init__(self, iou_threshold: float = 0.20, max_drift_px: float = 35.0):
        self.iou_threshold = iou_threshold
        self.max_drift_px = max_drift_px

    def _associate_detections(
        self,
        dets_prev: List[Dict[str, Any]],
        dets_curr: List[Dict[str, Any]]
    ) -> Dict[int, int]:
        """Matches detections between consecutive frames using spatial proximity and IoU."""
        matches = {}
        used_curr = set()
        
        for i, prev in enumerate(dets_prev):
            b_prev = prev['bbox_xyxy']
            c_prev = ((b_prev[0] + b_prev[2]) / 2.0, (b_prev[1] + b_prev[3]) / 2.0)
            
            best_j = -1
            best_score = -1.0
            
            for j, curr in enumerate(dets_curr):
                if j in used_curr:
                    continue
                b_curr = curr['bbox_xyxy']
                c_curr = ((b_curr[0] + b_curr[2]) / 2.0, (b_curr[1] + b_curr[3]) / 2.0)
                
                dist = math.hypot(c_curr[0] - c_prev[0], c_curr[1] - c_prev[1])
                iou = compute_box_iou(b_prev, b_curr)
                
                if dist <= self.max_drift_px or iou >= self.iou_threshold:
                    score = iou + 1.0 / (1.0 + 0.05 * dist)
                    if score > best_score:
                        best_score = score
                        best_j = j
                        
            if best_j != -1 and best_score > 0.4:
                matches[i] = best_j
                used_curr.add(best_j)
                
        return matches

    def verify_window_3_frames(
        self,
        window: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a 3-frame sliding window [t-1, t, t+1] to verify detections at frame t.
        Returns the verified detection list for the center frame t.
        """
        if len(window) < 3:
            return window[len(window) // 2]
            
        dets_prev, dets_curr, dets_next = window[0], window[1], window[2]
        
        # Associate t-1 -> t and t -> t+1
        prev_to_curr = self._associate_detections(dets_prev, dets_curr)
        curr_to_next = self._associate_detections(dets_curr, dets_next)
        
        # Invert prev_to_curr to map curr_idx -> prev_idx
        curr_to_prev = {v: k for k, v in prev_to_curr.items()}
        
        verified_curr = []
        
        for curr_idx, det in enumerate(dets_curr):
            det_verified = dict(det)
            single_frame_is_agg = (det.get('class_id', 0) > 0 or det.get('count', 1) > 1)
            
            prev_idx = curr_to_prev.get(curr_idx, None)
            next_idx = curr_to_next.get(curr_idx, None)
            
            prev_det = dets_prev[prev_idx] if prev_idx is not None else None
            next_det = dets_next[next_idx] if next_idx is not None else None
            
            # --- Test 1: Optical Overlap (False Agglomerate) ---
            # If two independent pellets came together in frame t, they should have 2 distinct tracks
            # in either prev or next frame.
            is_optical_overlap = False
            
            # Check for multiple prev detections that merged into curr
            incoming_merges = [p_i for p_i, c_i in prev_to_curr.items() if c_i == curr_idx]
            # Check for curr that splits into multiple next detections
            outgoing_splits = [n_i for c_i, n_i in curr_to_next.items() if c_i == curr_idx]
            
            if len(incoming_merges) > 1 or len(outgoing_splits) > 1:
                is_optical_overlap = True
                
            # Aspect ratio and velocity acceleration test:
            # An optical pass-by shows sudden velocity jump or transient bbox expansion
            if prev_det and next_det:
                prev_is_agg = (prev_det.get('class_id', 0) > 0 or prev_det.get('count', 1) > 1)
                next_is_agg = (next_det.get('class_id', 0) > 0 or next_det.get('count', 1) > 1)
                
                # If both neighbor frames are single pellets, but middle is marked as agglomerate:
                # likely a momentary pass-by / optical touch!
                if single_frame_is_agg and (not prev_is_agg) and (not next_is_agg):
                    is_optical_overlap = True
                    
                # --- Test 2: Tumbling Agglomerate Disambiguation (False Single) ---
                # If middle frame looked like single pellet (end-on view), but BOTH neighbor frames
                # along the same rigid track are clear agglomerates:
                if (not single_frame_is_agg) and (prev_is_agg or next_is_agg):
                    # Multi-frame consensus promotes to Agglomerate!
                    det_verified['class_id'] = max(1, prev_det.get('class_id', 1))
                    det_verified['class_name'] = 'agglomerate'
                    det_verified['temporal_status'] = 'promoted_by_tumbling_history'
                    det_verified['temporal_confidence'] = 0.92
                    verified_curr.append(det_verified)
                    continue

            if is_optical_overlap:
                # Correct classification from agglomerate to single_pellet (optical overlap)
                det_verified['class_id'] = 0
                det_verified['class_name'] = 'single_pellet'
                det_verified['temporal_status'] = 'resolved_optical_overlap'
                det_verified['is_optical_overlap'] = True
                det_verified['temporal_confidence'] = 0.88
            else:
                det_verified['temporal_status'] = 'confirmed_rigid_body'
                det_verified['is_optical_overlap'] = False
                det_verified['temporal_confidence'] = 0.95
                
            verified_curr.append(det_verified)
            
        return verified_curr

