import math
import random
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
    Features:
    - Multi-angle dynamic lighting (coaxial, side glancing, multi-azimuth LED banks)
    - Subsurface diffuse scattering ensuring smooth, natural meniscus necks (no black artifacts)
    - Dual specular reflections (sharp highlight + broad gloss sheen)
    - 3D surface normals with liquid-bridge meniscus necks for up to 20-pellet macro-clusters
    - Physically grounded Ambient Occlusion (AO) crevice shadows
    - Directional inter-sphere self-shadowing
    - Depth-of-field progressive defocus blur and velocity motion blur
    - Poisson-Gaussian camera sensor noise
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.view_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        
        # Current active lighting parameters
        self.light_dir = np.array(config.light_dir, dtype=float)
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.half_vector = (self.light_dir + self.view_dir) / np.linalg.norm(self.light_dir + self.view_dir)
        
        self.current_ambient = config.ambient_intensity
        self.current_diffuse = config.diffuse_intensity
        self.current_specular_broad = config.specular_broad
        self.current_specular_sharp = config.specular_sharp
        self.current_specular_sharp_exp = config.specular_sharp_exp
        self.current_lamp_gain = 1.0

    def sample_dynamic_lighting(self):
        """Randomizes photometric lighting parameters to create diverse illumination batches."""
        if not self.config.randomize_lighting:
            return
            
        cfg = self.config
        
        if cfg.lighting_mode == 'coaxial':
            # Direct top-down / front coaxial inspection ring light
            azimuth = random.uniform(0, 2 * math.pi)
            elevation = math.radians(random.uniform(70.0, 88.0))
        elif cfg.lighting_mode == 'side_grazing':
            # Low-angle glancing light to accentuate crevice edges
            azimuth = random.uniform(0, 2 * math.pi)
            elevation = math.radians(random.uniform(25.0, 45.0))
        elif cfg.lighting_mode == 'high_contrast':
            # Strong point source with sharp shadows
            azimuth = random.uniform(0, 2 * math.pi)
            elevation = math.radians(random.uniform(40.0, 65.0))
        else:
            # Dynamic multi-angle coverage
            azimuth = random.uniform(0, 2 * math.pi)
            elevation = math.radians(random.uniform(35.0, 82.0))
            
        lx = math.cos(elevation) * math.cos(azimuth)
        ly = math.cos(elevation) * math.sin(azimuth)
        lz = math.sin(elevation)
        
        self.light_dir = np.array([lx, ly, lz], dtype=float)
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.half_vector = (self.light_dir + self.view_dir) / np.linalg.norm(self.light_dir + self.view_dir)
        
        # Randomize intensities
        self.current_ambient = random.uniform(cfg.ambient_range[0], cfg.ambient_range[1])
        self.current_diffuse = random.uniform(cfg.diffuse_range[0], cfg.diffuse_range[1])
        self.current_specular_broad = random.uniform(cfg.specular_range[0], cfg.specular_range[1])
        self.current_specular_sharp = random.uniform(cfg.specular_range[0], cfg.specular_range[1])
        self.current_specular_sharp_exp = random.uniform(cfg.specular_exp_range[0], cfg.specular_exp_range[1])
        self.current_lamp_gain = random.uniform(cfg.lamp_drift_range[0], cfg.lamp_drift_range[1])

    def render_entity_patch(
        self,
        entity: PelletEntity
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Renders a local patch for a single pellet or macro-agglomerate (1 to 20 pellets).
        """
        transformed_nodes = entity.get_world_transformed_nodes()
        k_bridge = self.config.bridge_k
        
        xs_arr = [s[0] for s in transformed_nodes]
        ys_arr = [s[1] for s in transformed_nodes]
        max_r = max([s[3] for s in transformed_nodes])
        pad = int(max_r * 1.5) + 14
        
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
            
        edge_alpha = np.clip(-d_union + 0.5, 0.0, 1.0)
        
        # Surface normal computation with smooth liquid-bridge blending
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
            
            # Meniscus bridge region: smooth catenoid surface facing the camera
            bridge = (~in_sph) & (r2 <= (sr + k_bridge * 2.0)**2)
            dist_from_sph = np.sqrt(r2[bridge]) - sr
            dz[bridge] = np.maximum(0.0, sr * 0.50 - dist_from_sph * 0.40)
            
            n_len = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
            snx = dx / n_len
            sny = dy / n_len
            snz = np.maximum(0.35, dz / n_len)  # Enforce forward-facing camera normal on front surface
            
            dist = np.sqrt(r2)
            weight = np.exp(-0.5 * (dist / (sr * 0.7))**2) + 1e-5
            
            nx += snx * weight
            ny += sny * weight
            nz += snz * weight
            total_w += weight
            
        nx /= total_w
        ny /= total_w
        nz /= total_w
        
        # Guarantee that inside the pellet silhouette, the surface normal always points toward the camera
        nz = np.maximum(nz, 0.45)
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        nx /= norm
        ny /= norm
        nz /= norm
        
        # Physical Ambient Occlusion (AO) crevice shadows:
        # Real pellets have high subsurface scattering (albedo ~0.8), so contact valleys
        # are gentle soft shadows, NEVER black holes!
        ao = np.ones((h, w), dtype=float)
        N_nodes = len(transformed_nodes)
        if N_nodes > 1:
            for i in range(N_nodes):
                s1 = transformed_nodes[i]
                for j in range(i + 1, N_nodes):
                    s2 = transformed_nodes[j]
                    dist_2d = math.hypot(s1[0] - s2[0], s1[1] - s2[1])
                    if dist_2d > (s1[3] + s2[3]) * 1.30:
                        continue # Skip non-adjacent pairs
                        
                    c1 = np.array([s1[0], s1[1]])
                    c2 = np.array([s2[0], s2[1]])
                    seg = c2 - c1
                    seg_len = np.linalg.norm(seg)
                    if seg_len < 1e-4:
                        continue
                    seg_u = seg / seg_len
                    pts_x = xs - c1[0]
                    pts_y = ys - c1[1]
                    proj_t = (pts_x * seg_u[0] + pts_y * seg_u[1]) / seg_len
                    in_segment = (proj_t >= 0.15) & (proj_t <= 0.85)
                    
                    d1 = np.hypot(xs - s1[0], ys - s1[1])
                    d2 = np.hypot(xs - s2[0], ys - s2[1])
                    dist_from_spheres = np.sqrt((d1 - s1[3])**2 + (d2 - s2[3])**2)
                    crevice_factor = np.exp(-0.5 * (dist_from_spheres / self.config.ao_crevice_width)**2)
                    ao -= self.config.ao_crevice_strength * crevice_factor * in_segment
            # Gentle soft shadow floor: never drop below 0.72 (eliminates black neck artifacts)
            ao = np.clip(ao, 0.72, 1.0)
            
        # Directional inter-sphere self-shadowing (only between non-adjacent or distant spheres)
        shadow = np.ones((h, w), dtype=float)
        if N_nodes > 1:
            for idx_c, caster in enumerate(transformed_nodes):
                for idx_r, receiver in enumerate(transformed_nodes):
                    if idx_c == idx_r:
                        continue
                    caster_pos = np.array(caster[:3])
                    receiver_pos = np.array(receiver[:3])
                    rel = caster_pos - receiver_pos
                    dist_centers = math.hypot(rel[0], rel[1])
                    
                    # Connected adjacent spheres share a meniscus and do NOT cast hard shadows onto each other
                    if dist_centers < (caster[3] + receiver[3]) * 0.95:
                        continue
                    if dist_centers > (caster[3] + receiver[3]) * 2.5:
                        continue
                        
                    if np.dot(rel, self.light_dir) > 1.6:
                        dx_c = xs - (caster[0] + self.light_dir[0] * caster[3] * 0.6)
                        dy_c = ys - (caster[1] + self.light_dir[1] * caster[3] * 0.6)
                        sh_dist = np.hypot(dx_c, dy_c)
                        sh_factor = self.config.self_shadow_strength * np.clip(1.0 - (sh_dist / (caster[3] * 1.15)), 0.0, 1.0)
                        d_rec = np.hypot(xs - receiver[0], ys - receiver[1])
                        receiver_mask = d_rec <= receiver[3]
                        shadow[receiver_mask] *= (1.0 - sh_factor[receiver_mask])
            shadow = np.clip(shadow, 0.65, 1.0)

        # Dynamic Illumination model with Subsurface Scattering
        L = self.light_dir
        H = self.half_vector
        NdotL = np.maximum(0.0, nx * L[0] + ny * L[1] + nz * L[2])
        NdotH = np.maximum(0.0, nx * H[0] + ny * H[1] + nz * H[2])
        
        # Soft crevice shading with subsurface scattering floor
        ambient = self.current_ambient * (0.65 + 0.35 * ao)
        diffuse = self.current_diffuse * NdotL * (0.70 + 0.30 * ao) * shadow
        subsurface = 0.12 * (0.80 + 0.20 * ao)  # Internal diffuse scattering prevents pitch black valleys
        
        specular = (
            self.current_specular_broad * (NdotH ** self.config.specular_broad_exp) +
            self.current_specular_sharp * (NdotH ** self.config.specular_sharp_exp)
        ) * shadow
        
        # Micro-texture noise on pellet surface
        noise = np.random.normal(1.0, self.config.surface_noise_std, (h, w))
        intensity = (ambient + diffuse + subsurface + specular) * noise * 220.0
        intensity = np.clip(intensity, 0, 255.0)
        
        # Progressive Depth of Field (DoF)
        z_dist = abs(entity.position[2] - self.config.focus_z_center)
        excess_z = max(0.0, z_dist - self.config.focus_sharp_tolerance)
        defocus_sigma = excess_z * self.config.defocus_blur_rate
        attenuation = 1.0 / (1.0 + 0.035 * z_dist)
        intensity = intensity * attenuation
        
        fg = intensity * edge_alpha * attenuation
        fg_alpha = edge_alpha * attenuation
        
        if defocus_sigma > 0.4:
            ksize = int(round(defocus_sigma * 3.5)) | 1
            ksize = max(3, min(ksize, 25))
            fg = cv2.GaussianBlur(fg, (ksize, ksize), defocus_sigma)
            fg_alpha = cv2.GaussianBlur(fg_alpha, (ksize, ksize), defocus_sigma)
            
        # Velocity motion blur
        vx, vy = entity.velocity[0], entity.velocity[1]
        speed = math.hypot(vx, vy)
        if speed > 1.2:
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
        """
        self.sample_dynamic_lighting()
        
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
                            'morphology': entity.morphology,
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
                            global_pts[:, 0] = np.clip(global_pts[:, 0], 0, W_img - 1)
                            global_pts[:, 1] = np.clip(global_pts[:, 1], 0, H_img - 1)
                            annot['polygon'] = global_pts.tolist()
                            
                            norm_poly = []
                            for pt in global_pts:
                                norm_poly.extend([pt[0] / W_img, pt[1] / H_img])
                            annot['yolo_seg'] = norm_poly
                            
                        annotations.append(annot)
                        
        # Apply global lamp gain and sensor noise
        canvas = canvas * getattr(self, 'current_lamp_gain', 1.0)
        noise = np.random.normal(0, self.config.sensor_noise_std, canvas.shape)
        final_img = np.clip(canvas + noise, 0, 255.0).astype(np.uint8)
        
        return final_img, annotations
