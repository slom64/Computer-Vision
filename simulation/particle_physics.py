import math
import random
from typing import List, Optional
import numpy as np
from .config import SimulationConfig
from .pellet_models import PelletEntity

class FluidBedKinematics:
    """
    Simulates free 3D fluid-bed particle kinematics:
    - Central upward fluidization jet (spout/fountain flow)
    - Downward recirculation settling along outer walls
    - 3D tumbling rotation
    - Continuous steady-state spawning and despawning
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.entities: List[PelletEntity] = []
        self.next_id = 1
        self.initialize_population()
        
    def _sample_pellet_count(self) -> int:
        """Samples count of pellets for a new entity based on configured probabilities."""
        r = random.random()
        p1 = self.config.prob_single
        p2 = p1 + self.config.prob_dimer
        p3 = p2 + self.config.prob_trimer
        p4 = p3 + self.config.prob_cluster
        
        if r < p1:
            return 1
        elif r < p2:
            return 2
        elif r < p3:
            return 3
        elif r < p4:
            return random.randint(4, 6)
        else:
            return random.randint(7, self.config.max_cluster_size)

    def spawn_entity(
        self,
        initial_y: Optional[float] = None,
        is_spout_core: Optional[bool] = None
    ) -> PelletEntity:
        """Spawns a new pellet or agglomerate entity with physically plausible fluid-bed dynamics."""
        count = self._sample_pellet_count()
        entity = PelletEntity(
            entity_id=self.next_id,
            count=count,
            base_radius=self.config.base_radius,
            radius_variance=self.config.radius_variance,
            overlap_range=(self.config.overlap_min, self.config.overlap_max),
            morphology=self.config.cluster_morphology
        )
        self.next_id += 1
        
        cfg = self.config
        
        # Decide if this particle is in central spout or outer wall
        if is_spout_core is None:
            is_spout_core = (random.random() < 0.70)
            
        chamber_center_x = (cfg.chamber_x_min + cfg.chamber_x_max) / 2.0
        chamber_half_w = (cfg.chamber_x_max - cfg.chamber_x_min) / 2.0
        
        if is_spout_core:
            # Spawning in central upward fountain
            entity.position[0] = chamber_center_x + random.uniform(-chamber_half_w * 0.55, chamber_half_w * 0.55)
            entity.position[1] = initial_y if initial_y is not None else random.uniform(cfg.chamber_y_max - 50, cfg.chamber_y_max + 100)
            entity.position[2] = random.uniform(cfg.chamber_z_min, cfg.chamber_z_max)
            
            # Upward velocity (strongest in central column)
            dist_from_center = abs(entity.position[0] - chamber_center_x) / chamber_half_w
            core_factor = max(0.4, 1.0 - dist_from_center)
            up_vy = random.uniform(cfg.central_upward_velocity_min, cfg.central_upward_velocity_max) * core_factor
            
            # Slight outward lateral divergence
            vx_dir = 1.0 if entity.position[0] >= chamber_center_x else -1.0
            vx = vx_dir * random.uniform(0.5, 3.5) + np.random.normal(0, cfg.lateral_turbulent_std)
            
            entity.velocity = np.array([vx, up_vy, np.random.normal(0, 1.2)], dtype=float)
        else:
            # Spawning in outer downward recirculation wall
            side = random.choice([-1.0, 1.0])
            entity.position[0] = chamber_center_x + side * random.uniform(chamber_half_w * 0.60, chamber_half_w * 0.95)
            entity.position[1] = initial_y if initial_y is not None else random.uniform(cfg.chamber_y_min - 80, cfg.chamber_y_min + 50)
            entity.position[2] = random.uniform(cfg.chamber_z_min, cfg.chamber_z_max)
            
            # Settling velocity downward
            down_vy = random.uniform(cfg.wall_settling_velocity * 0.5, cfg.wall_settling_velocity * 1.5)
            vx = -side * random.uniform(0.2, 1.5) # Drift inward toward base
            entity.velocity = np.array([vx, down_vy, np.random.normal(0, 1.0)], dtype=float)
            
        # Random tumbling angular velocity
        entity.angular_velocity = np.random.normal(0, cfg.angular_tumbling_std, 3)
        return entity

    def initialize_population(self):
        """Initializes a steady-state particle swarm distributed across the view frustum."""
        self.entities.clear()
        cfg = self.config
        for _ in range(cfg.num_particles_active):
            rand_y = random.uniform(cfg.chamber_y_min, cfg.chamber_y_max)
            is_spout = (random.random() < 0.65)
            entity = self.spawn_entity(initial_y=rand_y, is_spout_core=is_spout)
            self.entities.append(entity)

    def step(self, dt: float = 1.0):
        """Advances physical kinematics by one frame step."""
        cfg = self.config
        chamber_center_x = (cfg.chamber_x_min + cfg.chamber_x_max) / 2.0
        chamber_half_w = (cfg.chamber_x_max - cfg.chamber_x_min) / 2.0
        
        surviving_entities = []
        for entity in self.entities:
            # Apply turbulence
            entity.velocity[0] += np.random.normal(0, 0.25 * cfg.lateral_turbulent_std)
            entity.velocity[1] += np.random.normal(0, 0.20)
            entity.velocity[2] += np.random.normal(0, 0.15)
            
            # Drag & buoyancy adjustment: particles near outer wall decelerate and fall
            dist_norm = abs(entity.position[0] - chamber_center_x) / chamber_half_w
            if dist_norm > 0.75:
                # Outside fountain core: downward gravity dominance
                entity.velocity[1] = min(entity.velocity[1] + 0.35, cfg.wall_settling_velocity * 1.4)
                
            entity.update_physics(dt)
            
            # Despawn checks
            out_top = entity.position[1] < (cfg.chamber_y_min - 120.0)
            out_bottom = entity.position[1] > (cfg.chamber_y_max + 120.0)
            out_left = entity.position[0] < (cfg.chamber_x_min - 100.0)
            out_right = entity.position[0] > (cfg.chamber_x_max + 100.0)
            out_z = abs(entity.position[2]) > (cfg.chamber_z_max + 8.0)
            
            if out_top or out_bottom or out_left or out_right or out_z:
                # Replace with newly spawned particle to maintain constant density
                surviving_entities.append(self.spawn_entity())
            else:
                surviving_entities.append(entity)
                
        self.entities = surviving_entities

