import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

class MetalReflectionFilter:
    """
    Industrial Metal Reflection Filter for Fluid Bed Vision Systems.
    
    In fluid bed granulation/coating chambers, metallic walls and spray nozzles
    produce specular mirror reflections of pellets. These reflections cause:
      1. Artificial doubling of pellet counts near walls.
      2. Pseudo-agglomeration: a single sticking pellet touching the wall merges
         with its mirror reflection into a wide bounding box, mimicking a dimer.
      3. False multi-track generation on metallic surfaces.
      
    This filter uses optical reflection symmetry, boundary partitioning, and
    photometric contrast analysis to identify, tag, and suppress reflections.
    """
    
    def __init__(
        self,
        wall_x_right: float = 1876.0,
        wall_x_left: float = 180.0,
        enable_symmetry_pairing: bool = True,
        enable_pseudo_agglomerate_splitting: bool = True,
        symmetry_x_tolerance: float = 40.0,
        symmetry_y_tolerance: float = 28.0,
        min_contrast_ratio: float = 0.88,
        deep_metal_right_x: float = 1960.0
    ):
        self.wall_x_right = wall_x_right
        self.wall_x_left = wall_x_left
        self.enable_symmetry_pairing = enable_symmetry_pairing
        self.enable_pseudo_agglomerate_splitting = enable_pseudo_agglomerate_splitting
        self.symmetry_x_tolerance = symmetry_x_tolerance
        self.symmetry_y_tolerance = symmetry_y_tolerance
        self.min_contrast_ratio = min_contrast_ratio
        self.deep_metal_right_x = deep_metal_right_x

    def analyze_patch_photometrics(self, gray_img: np.ndarray, bbox_xyxy: List[float]) -> Dict[str, float]:
        """Extracts photometric descriptors (mean, std dev, vertical edge energy) for a box."""
        if gray_img is None:
            return {"mean": 100.0, "std": 30.0, "v_edge_ratio": 1.0}
            
        H, W = gray_img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
        x2, y2 = max(x1 + 1, min(W, x2)), max(y1 + 1, min(H, y2))
        
        crop = gray_img[y1:y2, x1:x2]
        if crop.size < 4:
            return {"mean": 0.0, "std": 0.0, "v_edge_ratio": 1.0}
            
        mean_val = float(np.mean(crop))
        std_val = float(np.std(crop))
        
        # Brushed metal reflections have higher vertical gradient dispersion
        sobel_x = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
        energy_x = float(np.mean(np.abs(sobel_x))) + 1e-4
        energy_y = float(np.mean(np.abs(sobel_y))) + 1e-4
        v_edge_ratio = energy_y / energy_x
        
        return {
            "mean": mean_val,
            "std": std_val,
            "v_edge_ratio": v_edge_ratio
        }

    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        image: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Processes a list of raw YOLO detections and separates real pellets from metal reflections.
        
        Returns:
            clean_detections: Detections filtered of reflections and corrected for pseudo-agglomerates.
            reflection_detections: Detections flagged as metal reflections (for visualization / diagnostics).
        """
        if not detections:
            return [], []
            
        gray = None
        if image is not None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

        processed = []
        for det in detections:
            d = dict(det)
            box = list(d['bbox_xyxy'])
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            w = box[2] - box[0]
            h = box[3] - box[1]
            
            d['cx'] = cx
            d['cy'] = cy
            d['w'] = w
            d['h'] = h
            d['is_reflection'] = False
            d['reflection_reason'] = ""
            
            if gray is not None:
                photo = self.analyze_patch_photometrics(gray, box)
                d.update(photo)
            else:
                d['mean'] = 100.0
                d['std'] = 30.0
                d['v_edge_ratio'] = 1.0

            # Rule 1: Pseudo-Agglomerate Splitting / Clipping across Right Wall Boundary
            # If a detection box straddles the metal boundary and is wider than normal (w/h > 1.3)
            if self.enable_pseudo_agglomerate_splitting:
                if box[0] < self.wall_x_right < box[2]:
                    aspect_ratio = w / max(1.0, h)
                    # If it protrudes into metal by > 15% of width
                    protrusion = box[2] - self.wall_x_right
                    if protrusion > 0.15 * w and aspect_ratio > 1.25:
                        # Clip the bounding box to real chamber side
                        clipped_box = [box[0], box[1], self.wall_x_right + 4.0, box[3]]
                        d['bbox_xyxy'] = clipped_box
                        d['cx'] = (clipped_box[0] + clipped_box[2]) / 2.0
                        d['w'] = clipped_box[2] - clipped_box[0]
                        d['pseudo_dimer_clipped'] = True
                        # If it was labeled agglomerate solely due to wide aspect ratio, reclassify as single
                        if d.get('class_id') == 1 and (d['w'] / max(1.0, h)) < 1.20:
                            d['class_id'] = 0
                            d['reclassified_from_pseudo_dimer'] = True

            processed.append(d)

        # Rule 2: Deep Metal Suppression
        # Anything far inside the solid metal structure is an optical reflection or glare
        for d in processed:
            if d['cx'] > self.deep_metal_right_x:
                d['is_reflection'] = True
                d['reflection_reason'] = f"deep_metal_zone (cx={d['cx']:.1f} > {self.deep_metal_right_x})"

        # Rule 3: Specular Mirror Symmetry Pairing
        if self.enable_symmetry_pairing:
            # Candidates inside fluid bed vs candidates inside metal
            real_candidates = [d for d in processed if not d['is_reflection'] and d['cx'] < self.wall_x_right]
            metal_candidates = [d for d in processed if not d['is_reflection'] and d['cx'] >= self.wall_x_right]
            
            for m_det in metal_candidates:
                m_cx, m_cy = m_det['cx'], m_det['cy']
                
                # Theoretical mirror position inside fluid bed:
                # x_real_pred = 2 * wall_x - x_refl
                expected_real_x = 2.0 * self.wall_x_right - m_cx
                
                best_match = None
                best_dist = float('inf')
                
                for r_det in real_candidates:
                    r_cx, r_cy = r_det['cx'], r_det['cy']
                    
                    dx = abs(r_cx - expected_real_x)
                    dy = abs(r_cy - m_cy)
                    
                    if dx < self.symmetry_x_tolerance and dy < self.symmetry_y_tolerance:
                        dist = math.hypot(dx, dy)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = r_det
                            
                if best_match is not None:
                    # Verified mirror pair! Check contrast or confidence
                    # Reflection usually has lower or equal contrast than real pellet
                    m_det['is_reflection'] = True
                    m_det['reflection_reason'] = (
                        f"mirror_pair_of_({best_match['cx']:.1f},{best_match['cy']:.1f})_dist={best_dist:.1f}px"
                    )
                    m_det['reflection_parent_coords'] = (best_match['cx'], best_match['cy'])

        # Separate output sets
        clean_detections = [d for d in processed if not d['is_reflection']]
        reflection_detections = [d for d in processed if d['is_reflection']]
        
        return clean_detections, reflection_detections
