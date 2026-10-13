import math
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from .config import SimulationConfig
from .pellet_models import PelletEntity

def smin_val(a: np.ndarray, b: np.ndarray, k: float = 3.2) -> np.ndarray:
    """Polynomial smooth minimum for meniscus liquid-bridging between spheres."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25

class OpticalRenderer:
    """
    Photometric & Optical rendering engine calibrated to fluid bed inspection imaging.
    Simulates dual specular reflections, 3D surface normals, organic binder meniscus necks,
    depth-of-field defocus blur, directional motion blur, and camera sensor noise.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.light_dir = np.array(config.light_dir, dtype=float)
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.view_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        self.half_vector = (self.light_dir + self.view_dir) / np.linalg.norm(self.light_dir + self.view_dir)

    def render_entity_patch(
        self,
        entity: PelletEntity
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Renders a local patch for a single pellet or agglomerate entity.
        Returns:
            fg: Foreground intensity patch (float32)
            fg_alpha: Transparency alpha mask (float32)
            local_bbox: (x1, y1, x2, y2) bounding box relative to patch
            local_contour: Nx2 polygon contour for instance segmentation
            patch_bounds: (x_min, y_min, width, height) relative to entity center
        """
        transformed_nodes = entity.get_world_transformed_nodes()
        k_bridge = self.config.bridge_k
        
        xs_arr = [s[0] for s in transformed_nodes]
        ys_arr = [s[1] for s in transformed_nodes]
        max_r = max([s[3] for s in transformed_nodes])
        pad = int(max_r * 1.5) + 12
        
        x_min, x_max = int(min(xs_arr) - max_r - pad), int(max(xs_arr) + max_r + pad)
        y_min, y_max = int(min(ys_arr) - max_r - pad), int(max(ys_arr) + max_r + pad)
        
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        if w % 2 == 0:
            w += 1
        if h % 2 == 0:
            h += 1
            
        xs, ys = np.meshgrid(np.arange(x_min, x_min + w), np.arange(y_min, y_min + h))
        
        # 2D distance field with smooth minimum
        d_union = np.full((h, w), 999.0)
        for (sx, sy, sz, sr) in transformed_nodes:
            d_curr = np.hypot(xs - sx, ys - sy) - sr
            d_union = smin_val(d_union, d_curr, k_bridge)
            
        pellet_mask = d_union <= 0.0
        edge_alpha = np.clip(-d_union + 0.5, 0.0, 1.0)
        
        # Surface normal computation with sphere blending
        nx = np.zeros((h, w), dtype=float)
        ny = np.zeros((h, w), dtype=float)
        nz = np.zeros((h, w), dtype=float)
        total_w = np.zeros((h, w), dtype=float)
        
        for (sx, sy, sz, sr) in transformed_nodes:
            dx = xs - sx
            dy = ys - sy
            r2 = dx**2 + dy**2
            in_sph = r2 <= sr**2
            
            dz = np.zeros_like(dx)
            dz[in_sph] = np.sqrt(np.maximum(0, sr**2 - r2[in_sph]))
            bridge = (~in_sph) & (r2 <= (sr + k_bridge * 1.5)**2)
            dz[bridge] = np.maximum(0, sr * 0.35 - (np.sqrt(r2[bridge]) - sr) * 0.5)
            
            n_len = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
            snx = dx / n_len
            sny = dy / n_len
            snz = dz / n_len
            
            dist = np.sqrt(r2)
            weight = np.exp(-0.5 * (dist / (sr * 0.6))**2) + 1e-5
            
            nx += snx * weight
            ny += sny * weight
            nz += snz * weight
            total_w += weight
            
        nx /= total_w
        ny /= total_w
        nz /= total_w
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        nx /= norm
        ny /= norm
        nz /= norm
        
        # Illumination model
        L = self.light_dir
        H = self.half_vector
        NdotL = np.maximum(0, nx * L[0] + ny * L[1] + nz * L[2])
        NdotH = np.maximum(0, nx * H[0] + ny * H[1] + nz * H[2])
        
        ambient = self.config.ambient_intensity
        diffuse = self.config.diffuse_intensity * NdotL
        specular = (
            self.config.specular_broad * (NdotH ** self.config.specular_broad_exp) +
            self.config.specular_sharp * (NdotH ** self.config.specular_sharp_exp)
        )
        
        # Micro-texture noise on pellet surface
        noise = np.random.normal(1.0, self.config.surface_noise_std, (h, w))
        intensity = (ambient + diffuse + specular) * noise * 220.0
        intensity = np.clip(intensity, 0, 255.0)
        
        # Depth of Field (defocus blur & intensity attenuation)
        z_dist = abs(entity.position[2] - self.config.focus_z_center)
        defocus_sigma = self.config.defocus_base_sigma + self.config.defocus_blur_rate * z_dist
        attenuation = 1.0 / (1.0 + 0.04 * z_dist)
        intensity = intensity * attenuation
        
        fg = intensity * edge_alpha
        fg_alpha = edge_alpha * attenuation
        
        if defocus_sigma > 0.5:
            fg = cv2.GaussianBlur(fg, (0, 0), defocus_sigma)
            fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), defocus_sigma)
            
        # Directional motion blur
        vx, vy = entity.velocity[:2]
        speed = math.hypot(vx, vy)
        if speed > 1.5:
            kernel_len = int(min(speed * self.config.motion_blur_scale, self.config.max_motion_blur_kernel))
            if kernel_len >= 3:
                if kernel_len % 2 == 0:
                    kernel_len += 1
                angle = math.atan2(vy, vx)
                M = cv2.getRotationMatrix2D((kernel_len // 2, kernel_len // 2), math.degrees(angle), 1.0)
                k = np.zeros((kernel_len, kernel_len))
                k[kernel_len // 2, :] = 1.0 / kernel_len
                k = cv2.warpAffine(k, M, (kernel_len, kernel_len))
                k /= (k.sum() + 1e-6)
                fg = cv2.filter2D(fg, -1, k)
                fg_alpha = cv2.filter2D(fg_alpha, -1, k)
                
        # Local bounding box and polygon contour
        mask_thresh = fg_alpha > 0.18
        if np.any(mask_thresh):
            rows = np.where(mask_thresh.any(axis=1))[0]
            cols = np.where(mask_thresh.any(axis=0))[0]
            local_bbox = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
            
            mask_uint8 = (mask_thresh * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_c = max(contours, key=cv2.contourArea)
                local_contour = largest_c.reshape(-1, 2)
            else:
                local_contour = None
        else:
            local_bbox = None
            local_contour = None
            
        return fg, fg_alpha, local_bbox, local_contour, (x_min, y_min, w, h)

    def render_scene(
        self,
        background: np.ndarray,
        entities: List[PelletEntity]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Renders a full scene by compositing all entities onto the background plate.
        Returns:
            final_image: 2D uint8 image (H x W)
            annotations: List of ground-truth annotations for in-focus objects
        """
        H_img, W_img = background.shape[:2]
        canvas = background.astype(float).copy()
        
        # Depth sort: render far particles first, near particles in front
        sorted_entities = sorted(entities, key=lambda e: e.position[2], reverse=True)
        
        annotations = []
        
        for entity in sorted_entities:
            fg, alpha, lbox, lcontour, (x_min, y_min, pw, ph) = self.render_entity_patch(entity)
            
            dest_x1 = int(entity.position[0] + x_min)
            dest_y1 = int(entity.position[1] + y_min)
            dest_x2 = dest_x1 + pw
            dest_y2 = dest_y1 + ph
            
            src_x1 = max(0, -dest_x1)
            src_y1 = max(0, -dest_y1)
            src_x2 = pw - max(0, dest_x2 - W_img)
            src_y2 = ph - max(0, dest_y2 - H_img)
            
            dest_x1 = max(0, dest_x1)
            dest_y1 = max(0, dest_y1)
            dest_x2 = min(W_img, dest_x2)
            dest_y2 = min(H_img, dest_y2)
            
            if dest_x2 > dest_x1 and dest_y2 > dest_y1:
                sub_fg = fg[src_y1:src_y2, src_x1:src_x2]
                sub_alpha = alpha[src_y1:src_y2, src_x1:src_x2]
                
                curr_bg = canvas[dest_y1:dest_y2, dest_x1:dest_x2]
                canvas[dest_y1:dest_y2, dest_x1:dest_x2] = sub_fg + curr_bg * (1.0 - sub_alpha)
                
                # Check if in focus depth
                z_dist = abs(entity.position[2] - self.config.focus_z_center)
                if lbox is not None and z_dist <= self.config.focus_depth_tolerance:
                    gx1 = int(entity.position[0] + x_min + lbox[0])
                    gy1 = int(entity.position[1] + y_min + lbox[1])
                    gx2 = int(entity.position[0] + x_min + lbox[2])
                    gy2 = int(entity.position[1] + y_min + lbox[3])
                    
                    gx1 = max(0, min(W_img - 1, gx1))
                    gy1 = max(0, min(H_img - 1, gy1))
                    gx2 = max(0, min(W_img - 1, gx2))
                    gy2 = max(0, min(H_img - 1, gy2))
                    
                    if gx2 > gx1 and gy2 > gy1:
                        class_id = self.config.get_class_id(entity.count)
                        class_name = self.config.get_class_names()[class_id]
                        
                        annot = {
                            'id': entity.id,
                            'class_id': class_id,
                            'class_name': class_name,
                            'count': entity.count,
                            'type': entity.pellet_type.value,
                            'bbox_xyxy': [gx1, gy1, gx2, gy2],
                            'bbox_yolo': [
                                ((gx1 + gx2) / 2.0) / W_img,
                                ((gy1 + gy2) / 2.0) / H_img,
                                (gx2 - gx1) / W_img,
                                (gy2 - gy1) / H_img
                            ],
                            'position_3d': entity.position.tolist(),
                            'velocity': entity.velocity.tolist()
                        }
                        
                        if lcontour is not None:
                            global_pts = lcontour + np.array([int(entity.position[0] + x_min), int(entity.position[1] + y_min)])
                            # Clip polygon to image bounds
                            global_pts[:, 0] = np.clip(global_pts[:, 0], 0, W_img - 1)
                            global_pts[:, 1] = np.clip(global_pts[:, 1], 0, H_img - 1)
                            annot['polygon'] = global_pts.tolist()
                            
                            # Normalized polygon for YOLO segmentation
                            norm_poly = []
                            for pt in global_pts:
                                norm_poly.extend([pt[0] / W_img, pt[1] / H_img])
                            annot['yolo_seg'] = norm_poly
                            
                        annotations.append(annot)
                        
        # Camera sensor noise
        noise = np.random.normal(0, self.config.sensor_noise_std, canvas.shape)
        final_img = np.clip(canvas + noise, 0, 255.0).astype(np.uint8)
        
        return final_img, annotations

