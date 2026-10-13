import os
import glob
from typing import Optional, List
import cv2
import numpy as np
from .config import SimulationConfig

class BackgroundEngine:
    """
    Manages background plates for the simulation.
    Extracts static chamber plates from real dataset batches (e.g. Real-Data/T11/B4)
    using temporal median filtering, caches them, and applies subtle optical variations.
    Also provides a procedural chamber generator if no real images are supplied.
    """
    def __init__(self, config: SimulationConfig, cache_dir: str = 'simulation_cache'):
        self.config = config
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cached_plates: List[np.ndarray] = []
        
    def extract_from_real_data(
        self,
        real_data_dir: str = 'Real-Data/T11/B4',
        max_frames: int = 20,
        cache_name: str = 'bg_plate_T11_B4.png'
    ) -> np.ndarray:
        """
        Computes a clean static background plate from real camera images using
        temporal median filtering to eliminate fast-moving foreground pellets.
        """
        cache_path = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(cache_path):
            plate = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
            if plate is not None and plate.shape == (self.config.image_height, self.config.image_width):
                if not any(np.array_equal(plate, p) for p in self.cached_plates):
                    self.cached_plates.append(plate)
                return plate
                
        bmp_files = sorted(glob.glob(os.path.join(real_data_dir, '*.bmp')))
        if not bmp_files:
            bmp_files = sorted(glob.glob(os.path.join(real_data_dir, '**', '*.bmp'), recursive=True))
            
        if not bmp_files:
            print(f"Warning: No .bmp images found in {real_data_dir}. Falling back to procedural background.")
            return self.generate_procedural_background()
            
        selected_files = bmp_files[:max_frames]
        frames = []
        for f in selected_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (self.config.image_height, self.config.image_width):
                    img = cv2.resize(img, (self.config.image_width, self.config.image_height))
                frames.append(img)
                
        if len(frames) == 0:
            return self.generate_procedural_background()
            
        # Median across frames removes transient moving particles
        median_bg = np.median(frames, axis=0).astype(np.uint8)
        cv2.imwrite(cache_path, median_bg)
        self.cached_plates.append(median_bg)
        return median_bg

    def generate_procedural_background(self) -> np.ndarray:
        """
        Generates a synthetic fluid bed chamber background with metallic side nozzle,
        bottom curved flange, intense illumination glare bloom, and dense out-of-focus bokeh swarm.
        """
        H = self.config.image_height
        W = self.config.image_width
        bg = np.zeros((H, W), dtype=np.float32)
        
        # Base ambient gradient (dark top-left to slightly lighter center)
        ys, xs = np.ogrid[:H, :W]
        bg += 28.0 + 12.0 * np.sin(ys / H * np.pi * 0.7)
        
        # Static bright illuminator flare on glass (x ~ 450, y ~ 850)
        flare_x, flare_y = int(W * 0.23), int(H * 0.55)
        dist_flare = np.sqrt(((xs - flare_x) / 120.0)**2 + ((ys - flare_y) / 280.0)**2)
        flare = 255.0 / (1.0 + (dist_flare * 1.8)**2.2)
        bg = np.maximum(bg, flare)
        
        # Dense background bokeh particles (deep out-of-focus swarm)
        np.random.seed(101)
        num_bokeh = 2200
        bokeh_canvas = np.zeros((H, W), dtype=np.float32)
        for _ in range(num_bokeh):
            bx = np.random.uniform(50, W - 150)
            by = np.random.uniform(50, H - 200)
            br = np.random.uniform(6.0, 28.0)
            b_int = np.random.uniform(15.0, 65.0)
            cv2.circle(bokeh_canvas, (int(bx), int(by)), int(br), float(b_int), -1)
            
        bokeh_canvas = cv2.GaussianBlur(bokeh_canvas, (31, 31), 9.0)
        bg = np.clip(bg + bokeh_canvas, 0, 255)
        
        # Metallic right nozzle column
        metal_mask = np.zeros((H, W), dtype=np.uint8)
        nozzle_left = int(W * 0.75)
        pts_nozzle = np.array([
            [nozzle_left, int(H * 0.45)],
            [int(W * 0.90), int(H * 0.40)],
            [W, int(H * 0.40)],
            [W, H],
            [nozzle_left, H]
        ], dtype=np.int32)
        cv2.fillPoly(metal_mask, [pts_nozzle], 255)
        
        metal_tex = np.full((H, W), 45.0, dtype=np.float32)
        # Brushed metal vertical streaks
        streaks = np.random.normal(1.0, 0.08, (1, W))
        streaks = cv2.resize(streaks, (W, H))
        metal_tex *= streaks
        
        bg = np.where(metal_mask > 0, metal_tex, bg)
        
        # Bottom curved flange lip
        lip_y = int(H * 0.82)
        cv2.line(bg, (0, lip_y), (nozzle_left, lip_y + 15), 180.0, 3)
        cv2.line(bg, (0, lip_y + 3), (nozzle_left, lip_y + 18), 70.0, 2)
        
        # Add slight sensor noise
        bg += np.random.normal(0, 1.5, bg.shape)
        return np.clip(bg, 0, 255).astype(np.uint8)

    def get_background(self, jitter_brightness: bool = True) -> np.ndarray:
        """Returns an active background plate with optional subtle brightness jitter."""
        if len(self.cached_plates) == 0:
            plate = self.extract_from_real_data()
        else:
            plate = self.cached_plates[0].copy()
            
        if jitter_brightness:
            # Subtle lamp flicker / drift (+/- 3%)
            factor = np.random.uniform(0.97, 1.03)
            plate = np.clip(plate.astype(float) * factor, 0, 255).astype(np.uint8)
            
        return plate

