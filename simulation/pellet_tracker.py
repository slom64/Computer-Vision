import math
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4

class KalmanBoxTracker:
    """
    Kalman Filter for tracking individual pellets in 2D image coordinates.
    State vector: [x_c, y_c, s, r, vx, vy, vs]
    where (x_c, y_c) is bounding box center, s is scale (area = w * h),
    r is aspect ratio (w / h), and (vx, vy, vs) are first derivatives.
    """
    count = 0

    def __init__(self, bbox_xyxy: List[float], score: float = 1.0, class_id: int = 0):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        
        # State dimension: 7, Measurement dimension: 4
        self.dim_x = 7
        self.dim_z = 4
        
        # Initial state estimate
        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = self._xyxy_to_z(bbox_xyxy)
        
        # State transition matrix F (constant velocity model)
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1.0  # x_c += vx
        self.F[1, 5] = 1.0  # y_c += vy
        self.F[2, 6] = 1.0  # s += vs
        
        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(self.dim_z)
        
        # Covariance matrices tuned for high-speed fluid-bed pellets
        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty in initial velocities
        
        # Process noise covariance Q
        self.Q = np.eye(self.dim_x)
        self.Q[0:2, 0:2] *= 1.5   # Positional air turbulence
        self.Q[2, 2] *= 0.1       # Scale changes slowly
        self.Q[3, 3] *= 0.05      # Aspect ratio changes slowly
        self.Q[4:6, 4:6] *= 4.0   # Velocity variance from jet
        self.Q[6, 6] *= 0.1
        
        # Measurement noise covariance R
        self.R = np.eye(self.dim_z)
        self.R[0:2, 0:2] *= 1.0   # Center measurement precision
        self.R[2, 2] *= 8.0       # Scale measurement precision
        self.R[3, 3] *= 2.0       # Aspect ratio precision
        
        # Track statistics & state
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.state = TrackState.TENTATIVE
        
        # History buffers
        self.history_boxes: List[List[float]] = [bbox_xyxy]
        self.history_scores: List[float] = [score]
        self.history_classes: List[int] = [class_id]
        self.history_crops: List[np.ndarray] = []
        
        # Sequence classification & verification results
        self.sequence_class_id: int = class_id
        self.sequence_confidence: float = score
        self.is_optical_overlap: bool = False
        self.classification_notes: str = "single_frame_init"

    @staticmethod
    def _xyxy_to_z(bbox: List[float]) -> np.ndarray:
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        xc = bbox[0] + w / 2.0
        yc = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([[xc], [yc], [s], [r]])

    @staticmethod
    def _z_to_xyxy(z: np.ndarray) -> List[float]:
        xc, yc, s, r = z[0, 0], z[1, 0], max(1.0, z[2, 0]), max(0.1, z[3, 0])
        w = math.sqrt(s * r)
        h = s / w
        return [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0]

    def predict(self) -> List[float]:
        """Advances state vector using constant velocity model."""
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state_xyxy()

    def update(self, bbox_xyxy: List[float], score: float, class_id: int):
        """Updates Kalman Filter with measured bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        z = self._xyxy_to_z(bbox_xyxy)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman Gain
        
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
        
        self.history_boxes.append(bbox_xyxy)
        self.history_scores.append(score)
        self.history_classes.append(class_id)
        
        if len(self.history_boxes) > 60:
            self.history_boxes.pop(0)
            self.history_scores.pop(0)
            self.history_classes.pop(0)

    def append_crop(self, crop: np.ndarray, max_len: int = 16):
        """Maintains fixed-size rolling buffer of normalized particle crops."""
        self.history_crops.append(crop)
        if len(self.history_crops) > max_len:
            self.history_crops.pop(0)

    def get_state_xyxy(self) -> List[float]:
        return self._z_to_xyxy(self.x)

    def get_velocity(self) -> Tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])


def compute_iou_matrix(boxesA: List[List[float]], boxesB: List[List[float]]) -> np.ndarray:
    """Computes IoU cost matrix between two sets of bounding boxes."""
    if len(boxesA) == 0 or len(boxesB) == 0:
        return np.zeros((len(boxesA), len(boxesB)), dtype=float)
        
    A = np.array(boxesA, dtype=float)
    B = np.array(boxesB, dtype=float)
    
    xA = np.maximum(A[:, 0:1], B[:, 0:1].T)
    yA = np.maximum(A[:, 1:2], B[:, 1:2].T)
    xB = np.minimum(A[:, 2:3], B[:, 2:3].T)
    yB = np.minimum(A[:, 3:4], B[:, 3:4].T)
    
    interArea = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxAArea = np.maximum(1.0, (A[:, 2:3] - A[:, 0:1]) * (A[:, 3:4] - A[:, 1:2]))
    boxBArea = np.maximum(1.0, (B[:, 2:3] - B[:, 0:1]) * (B[:, 3:4] - B[:, 1:2]))
    
    union = boxAArea + boxBArea.T - interArea
    return interArea / np.maximum(1.0, union)


class PelletTracker:
    """
    ByteTrack Multi-Object Tracker tailored for fluid bed pellets.
    Tracks pellets across frames, recovers low-confidence / blurry pellets,
    extracts rolling crop trajectories, and feeds them to sequence models.
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.35,
        low_conf_thresh: float = 0.12,
        iou_thresh: float = 0.22,
        max_lost_frames: int = 6,
        min_hits_to_confirm: int = 2,
        crop_size: int = 96,
        max_crop_history: int = 16
    ):
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.iou_thresh = iou_thresh
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.crop_size = crop_size
        self.max_crop_history = max_crop_history
        
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def extract_crop(self, image: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
        """
        Extracts a centered, square crop of size (crop_size, crop_size) around pellet.
        Applies symmetric padding if the bounding box extends outside image boundaries.
        """
        H, W = image.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_s = self.crop_size / 2.0
        
        src_x1 = int(round(cx - half_s))
        src_y1 = int(round(cy - half_s))
        src_x2 = src_x1 + self.crop_size
        src_y2 = src_y1 + self.crop_size
        
        pad_left = max(0, -src_x1)
        pad_top = max(0, -src_y1)
        pad_right = max(0, src_x2 - W)
        pad_bottom = max(0, src_y2 - H)
        
        clamped_x1 = max(0, src_x1)
        clamped_y1 = max(0, src_y1)
        clamped_x2 = min(W, src_x2)
        clamped_y2 = min(H, src_y2)
        
        sub_img = image[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
        
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            sub_img = cv2.copyMakeBorder(
                sub_img, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )
            
        if sub_img.shape[:2] != (self.crop_size, self.crop_size):
            sub_img = cv2.resize(sub_img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            
        return sub_img

    def update(
        self,
        detections: List[Dict[str, Any]],
        image: Optional[np.ndarray] = None
    ) -> List[KalmanBoxTracker]:
        """
        Performs ByteTrack association on new detections for the current frame.
        detections: List of dicts containing:
            'bbox_xyxy': [x1, y1, x2, y2],
            'confidence': float,
            'class_id': int
        """
        self.frame_count += 1
        
        # 1. Predict all current tracks with Kalman Filter
        for trk in self.tracks:
            trk.predict()
            
        # 2. Partition detections into High-Score and Low-Score sets
        dets_high = []
        dets_low = []
        for det in detections:
            conf = det.get('confidence', 1.0)
            if conf >= self.high_conf_thresh:
                dets_high.append(det)
            elif conf >= self.low_conf_thresh:
                dets_low.append(det)
                
        # ---------------- Stage 1: Match Tracks with High-Score Detections ----------------
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_high = self._associate(
            self.tracks, dets_high, self.iou_thresh
        )
        
        for trk_idx, det_idx in matched_tracks_1:
            det = dets_high[det_idx]
            self.tracks[trk_idx].update(det['bbox_xyxy'], det.get('confidence', 1.0), det.get('class_id', 0))
            if image is not None:
                crop = self.extract_crop(image, det['bbox_xyxy'])
                self.tracks[trk_idx].append_crop(crop, self.max_crop_history)
                
        # ---------------- Stage 2: Match Unmatched Tracks with Low-Score Detections ----------------
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks_1]
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            remaining_tracks, dets_low, self.iou_thresh * 0.75
        )
        
        for rem_trk_idx, det_idx in matched_tracks_2:
            det = dets_low[det_idx]
            remaining_tracks[rem_trk_idx].update(det['bbox_xyxy'], det.get('confidence', 0.5), det.get('class_id', 0))
            if image is not None:
                crop = self.extract_crop(image, det['bbox_xyxy'])
                remaining_tracks[rem_trk_idx].append_crop(crop, self.max_crop_history)
                
        # ---------------- Stage 3: Handle Unmatched Tracks & New Detections ----------------
        # Truly unmatched tracks
        lost_track_indices = [unmatched_tracks_1[i] for i in unmatched_tracks_2]
        for trk_idx in lost_track_indices:
            trk = self.tracks[trk_idx]
            trk.hit_streak = 0
            if trk.state == TrackState.CONFIRMED:
                trk.state = TrackState.LOST
                
        # Create new tentative tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets_high:
            det = dets_high[det_idx]
            new_trk = KalmanBoxTracker(det['bbox_xyxy'], det.get('confidence', 1.0), det.get('class_id', 0))
            if image is not None:
                crop = self.extract_crop(image, det['bbox_xyxy'])
                new_trk.append_crop(crop, self.max_crop_history)
            self.tracks.append(new_trk)
            
        # ---------------- State Transitions & Pruning ----------------
        surviving_tracks = []
        for trk in self.tracks:
            # Upgrade tentative tracks that consistently matched
            if trk.state == TrackState.TENTATIVE and trk.hits >= self.min_hits_to_confirm:
                trk.state = TrackState.CONFIRMED
                
            # Remove tracks that have been lost too long
            if trk.time_since_update > self.max_lost_frames:
                trk.state = TrackState.DELETED
            else:
                surviving_tracks.append(trk)
                
        self.tracks = surviving_tracks
        
        # Return currently active tracks
        return [t for t in self.tracks if t.state in [TrackState.CONFIRMED, TrackState.TENTATIVE] and t.time_since_update <= 1]

    def _associate(
        self,
        track_list: List[KalmanBoxTracker],
        detection_list: List[Dict[str, Any]],
        iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Matches tracks to detections using Hungarian bipartite matching on IoU matrix."""
        if len(track_list) == 0 or len(detection_list) == 0:
            return [], list(range(len(track_list))), list(range(len(detection_list)))
            
        trk_boxes = [t.get_state_xyxy() for t in track_list]
        det_boxes = [d['bbox_xyxy'] for d in detection_list]
        
        iou_matrix = compute_iou_matrix(trk_boxes, det_boxes)
        
        # Linear sum assignment minimizes cost -> cost = -IoU
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched_pairs = []
        unmatched_tracks = set(range(len(track_list)))
        unmatched_dets = set(range(len(detection_list)))
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matched_pairs.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)
                
        return matched_pairs, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))

    def reset(self):
        KalmanBoxTracker.count = 0
        self.tracks.clear()
        self.frame_count = 0

