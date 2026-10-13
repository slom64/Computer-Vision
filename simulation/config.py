from typing import Tuple, List, Dict, Any
import numpy as np

class SimulationConfig:
    """Master configuration for fluid bed pellet simulation & dataset generation."""
    
    def __init__(
        self,
        # Image & Optical Sensor
        image_width: int = 2048,
        image_height: int = 1536,
        output_grayscale: bool = True,
        
        # Pellet Physical Geometry
        base_radius: float = 13.0,
        radius_variance: float = 0.14,
        aspect_ratio_min: float = 0.92,
        aspect_ratio_max: float = 1.08,
        bridge_k: float = 3.2,
        overlap_min: float = 0.80,
        overlap_max: float = 0.92,
        
        # Population & Agglomerate Distribution
        num_particles_active: int = 240,
        prob_single: float = 0.30,
        prob_dimer: float = 0.22,
        prob_trimer: float = 0.16,
        prob_cluster: float = 0.18,
        prob_macro: float = 0.14,
        max_cluster_size: int = 20,
        cluster_morphology: str = 'mixed',
        
        # Dynamic Lighting & Shading Engine
        lighting_mode: str = 'dynamic',
        randomize_lighting: bool = True,
        light_dir: Tuple[float, float, float] = (0.15, -0.22, 0.95),
        ambient_intensity: float = 0.34,
        diffuse_intensity: float = 0.44,
        specular_broad: float = 0.26,
        specular_sharp: float = 0.26,
        specular_broad_exp: float = 5.0,
        specular_sharp_exp: float = 20.0,
        
        # Dynamic lighting sampling ranges
        ambient_range: Tuple[float, float] = (0.22, 0.42),
        diffuse_range: Tuple[float, float] = (0.30, 0.52),
        specular_range: Tuple[float, float] = (0.16, 0.40),
        specular_exp_range: Tuple[float, float] = (4.0, 28.0),
        lamp_drift_range: Tuple[float, float] = (0.88, 1.15),
        
        surface_noise_std: float = 0.040,
        sensor_noise_std: float = 1.8,
        ao_crevice_strength: float = 0.22,  # Calibrated soft valley crevice (no black artifacts)
        ao_crevice_width: float = 3.5,
        self_shadow_strength: float = 0.35,
        
        # Optics: Depth of Field & Motion Blur
        focus_z_center: float = 0.0,
        focus_sharp_tolerance: float = 1.8,
        focus_depth_tolerance: float = 5.5,
        defocus_blur_rate: float = 0.22,
        motion_blur_scale: float = 0.30,
        max_motion_blur_kernel: int = 13,
        
        # Fluid Bed Kinematics
        chamber_x_min: float = 180.0,
        chamber_x_max: float = 1720.0,
        chamber_y_min: float = 120.0,
        chamber_y_max: float = 1280.0,
        chamber_z_min: float = -14.0,
        chamber_z_max: float = 14.0,
        
        central_upward_velocity_min: float = -12.0,
        central_upward_velocity_max: float = -5.0,
        wall_settling_velocity: float = 3.5,
        lateral_turbulent_std: float = 1.8,
        angular_tumbling_std: float = 0.16,
        
        # Classification mode: 'binary' or 'detailed'
        classification_mode: str = 'binary',
        
        # Gracefully handle any unexpected keyword arguments
        **kwargs: Any
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.output_grayscale = output_grayscale
        
        self.base_radius = base_radius
        self.radius_variance = radius_variance
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.bridge_k = bridge_k
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max
        
        self.num_particles_active = num_particles_active
        self.prob_single = prob_single
        self.prob_dimer = prob_dimer
        self.prob_trimer = prob_trimer
        self.prob_cluster = prob_cluster
        self.prob_macro = prob_macro
        self.max_cluster_size = max_cluster_size
        self.cluster_morphology = cluster_morphology
        
        self.lighting_mode = lighting_mode
        self.randomize_lighting = randomize_lighting
        self.light_dir = light_dir
        self.ambient_intensity = ambient_intensity
        self.diffuse_intensity = diffuse_intensity
        self.specular_broad = specular_broad
        self.specular_sharp = specular_sharp
        self.specular_broad_exp = specular_broad_exp
        self.specular_sharp_exp = specular_sharp_exp
        
        self.ambient_range = ambient_range
        self.diffuse_range = diffuse_range
        self.specular_range = specular_range
        self.specular_exp_range = specular_exp_range
        self.lamp_drift_range = lamp_drift_range
        
        self.surface_noise_std = surface_noise_std
        self.sensor_noise_std = sensor_noise_std
        self.ao_crevice_strength = ao_crevice_strength
        self.ao_crevice_width = ao_crevice_width
        self.self_shadow_strength = self_shadow_strength
        
        self.focus_z_center = focus_z_center
        self.focus_sharp_tolerance = focus_sharp_tolerance
        self.focus_depth_tolerance = focus_depth_tolerance
        self.defocus_blur_rate = defocus_blur_rate
        self.motion_blur_scale = motion_blur_scale
        self.max_motion_blur_kernel = max_motion_blur_kernel
        
        self.chamber_x_min = chamber_x_min
        self.chamber_x_max = chamber_x_max
        self.chamber_y_min = chamber_y_min
        self.chamber_y_max = chamber_y_max
        self.chamber_z_min = chamber_z_min
        self.chamber_z_max = chamber_z_max
        
        self.central_upward_velocity_min = central_upward_velocity_min
        self.central_upward_velocity_max = central_upward_velocity_max
        self.wall_settling_velocity = wall_settling_velocity
        self.lateral_turbulent_std = lateral_turbulent_std
        self.angular_tumbling_std = angular_tumbling_std
        
        self.classification_mode = classification_mode
        
        # Store any additional kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def get_class_names(self) -> Dict[int, str]:
        if self.classification_mode == 'binary':
            return {0: 'single_pellet', 1: 'agglomerate'}
        else:
            return {
                0: 'single_pellet',
                1: 'agglomerate_2',
                2: 'agglomerate_3',
                3: 'agglomerate_cluster',
                4: 'agglomerate_macro'
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
            elif count <= 6:
                return 3
            else:
                return 4
