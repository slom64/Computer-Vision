from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import numpy as np

@dataclass
class SimulationConfig:
    """Master configuration for fluid bed pellet simulation & dataset generation."""
    
    # ---------------- Image & Optical Sensor ----------------
    image_width: int = 2048
    image_height: int = 1536
    output_grayscale: bool = True       # Sensor outputs monochrome (saved as 3-ch identical BGR or 1-ch)
    
    # ---------------- Pellet Physical Geometry ----------------
    base_radius: float = 13.0          # Average radius in pixels matching real camera sensor
    radius_variance: float = 0.14      # Relative variation (+/- 14% to cover small and large pellets)
    aspect_ratio_min: float = 0.92     # Natural non-sphericity
    aspect_ratio_max: float = 1.08
    bridge_k: float = 3.2              # Meniscus liquid-bridge blending between agglomerated spheres
    overlap_min: float = 0.80          # Center distance ratio for agglomeration bonds
    overlap_max: float = 0.92
    
    # ---------------- Population & Class Distribution ----------------
    # Controls how many particles exist simultaneously in the chamber
    num_particles_active: int = 240    # Dense swarm matching real fluid bed chamber (200-350 pellets)

    
    # Probability distribution of particle types when spawned
    # (single: 1 pellet, dimer: 2, trimer: 3, cluster: 4 to 6)
    prob_single: float = 0.40
    prob_dimer: float = 0.30
    prob_trimer: float = 0.18
    prob_cluster: float = 0.12
    max_cluster_size: int = 6
    
    # ---------------- Lighting & Shading Model ----------------
    light_dir: Tuple[float, float, float] = (0.15, -0.22, 0.95) # Top-front light source
    ambient_intensity: float = 0.32
    diffuse_intensity: float = 0.42
    specular_broad: float = 0.28
    specular_sharp: float = 0.28
    specular_broad_exp: float = 5.0
    specular_sharp_exp: float = 22.0
    surface_noise_std: float = 0.045   # Micro-roughness on pellet surface
    sensor_noise_std: float = 1.8      # Poisson-Gaussian camera noise floor
    ao_crevice_strength: float = 0.65  # Ambient occlusion depth in neck valleys
    ao_crevice_width: float = 3.0      # Valley width in pixels
    self_shadow_strength: float = 0.50 # Directional inter-sphere shadowing
    
    # ---------------- Optics: Depth of Field & Motion Blur ----------------
    focus_z_center: float = 0.0        # Focal plane location
    focus_sharp_tolerance: float = 1.8 # Particles within |z| < 1.8 are crystal-sharp (zero defocus blur)
    focus_depth_tolerance: float = 5.5 # Particles within |z| < tolerance are considered in-focus labels
    defocus_blur_rate: float = 0.22    # Growth rate of progressive defocus blur per unit z-distance
    motion_blur_scale: float = 0.30    # Exposure time velocity blur scaling factor
    max_motion_blur_kernel: int = 13   # Max kernel size in pixels
    
    # ---------------- Fluid Bed Kinematics ----------------
    # Viewable chamber region (excluding static metal nozzle on right and bottom flange)
    chamber_x_min: float = 180.0
    chamber_x_max: float = 1720.0
    chamber_y_min: float = 120.0
    chamber_y_max: float = 1280.0
    chamber_z_min: float = -14.0
    chamber_z_max: float = 14.0
    
    # Velocity parameters (pixels per frame at ~30-60 fps)
    central_upward_velocity_min: float = -12.0 # Negative Y is upward
    central_upward_velocity_max: float = -5.0
    wall_settling_velocity: float = 3.5        # Downward settling along outer walls
    lateral_turbulent_std: float = 1.8         # Brownian air fluctuations
    angular_tumbling_std: float = 0.16         # 3D tumbling rate in radians/frame
    
    # ---------------- Annotation & Dataset Export ----------------
    # Classification mode:
    # 'binary': 0 = Single Pellet, 1 = Agglomerate
    # 'detailed': 0 = Single, 1 = Dimer (2), 2 = Trimer (3), 3 = Multimer (4+)
    classification_mode: str = 'detailed'
    
    def get_class_names(self) -> Dict[int, str]:
        if self.classification_mode == 'binary':
            return {0: 'single_pellet', 1: 'agglomerate'}
        else:
            return {
                0: 'single_pellet',
                1: 'agglomerate_2',
                2: 'agglomerate_3',
                3: 'agglomerate_cluster'
            }
            
    def get_class_id(self, count: int) -> int:
        if self.classification_mode == 'binary':
            return 0 if count == 1 else 1
        else:
            if count == 1:
                return 0
            elif count == 2:
                return 1
            elif count == 3:
                return 2
            else:
                return 3

