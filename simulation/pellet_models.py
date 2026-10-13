import math
import random
from enum import Enum
from typing import List, Tuple
import numpy as np

class PelletType(Enum):
    SINGLE = "single"
    DIMER = "agglomerate_2"
    TRIMER = "agglomerate_3"
    CLUSTER = "agglomerate_cluster"

class PelletEntity:
    """
    Represents an individual physical entity in 3D fluid-bed space:
    either a single pellet or an agglomerate of multiple connected spheres.
    """
    def __init__(
        self,
        entity_id: int,
        count: int,
        base_radius: float = 13.5,
        radius_variance: float = 0.08,
        overlap_range: Tuple[float, float] = (0.80, 0.92)
    ):
        self.id = entity_id
        self.count = count
        self.base_radius = base_radius
        
        # Categorize
        if count == 1:
            self.pellet_type = PelletType.SINGLE
        elif count == 2:
            self.pellet_type = PelletType.DIMER
        elif count == 3:
            self.pellet_type = PelletType.TRIMER
        else:
            self.pellet_type = PelletType.CLUSTER
            
        # 3D spatial state
        self.position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # 3D rotation representation: Euler angles and rotation matrix
        self.euler = np.array([
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi)
        ], dtype=float)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.rotation_matrix = self._compute_rotation_matrix()
        
        # Build 3D sphere nodes: list of (x, y, z, r) relative to center of mass
        self.nodes = self._generate_nodes(radius_variance, overlap_range)
        self._center_nodes_at_com()
        
    def _compute_rotation_matrix(self) -> np.ndarray:
        rx, ry, rz = self.euler
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _generate_nodes(
        self,
        radius_variance: float,
        overlap_range: Tuple[float, float]
    ) -> List[Tuple[float, float, float, float]]:
        """Generates 3D constituent spherical nodes for single or agglomerated pellets."""
        nodes = []
        r0 = self.base_radius * random.uniform(1.0 - radius_variance, 1.0 + radius_variance)
        nodes.append((0.0, 0.0, 0.0, r0))
        
        if self.count == 1:
            return nodes
            
        # Build agglomerate tree
        for _ in range(self.count - 1):
            parent = random.choice(nodes)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0.15 * math.pi, 0.85 * math.pi)
            
            r_child = self.base_radius * random.uniform(1.0 - radius_variance, 1.0 + radius_variance)
            overlap = random.uniform(overlap_range[0], overlap_range[1])
            dist = (parent[3] + r_child) * overlap
            
            dx = dist * math.sin(phi) * math.cos(theta)
            dy = dist * math.sin(phi) * math.sin(theta)
            dz = dist * math.cos(phi)
            
            nodes.append((parent[0] + dx, parent[1] + dy, parent[2] + dz, r_child))
            
        return nodes

    def _center_nodes_at_com(self):
        """Centers the local nodes at their center of mass for realistic tumbling."""
        xs = [n[0] for n in self.nodes]
        ys = [n[1] for n in self.nodes]
        zs = [n[2] for n in self.nodes]
        com_x = sum(xs) / len(xs)
        com_y = sum(ys) / len(ys)
        com_z = sum(zs) / len(zs)
        
        self.nodes = [
            (n[0] - com_x, n[1] - com_y, n[2] - com_z, n[3])
            for n in self.nodes
        ]

    def update_physics(self, dt: float = 1.0):
        """Step position and 3D tumbling rotation."""
        self.position += self.velocity * dt
        self.euler = (self.euler + self.angular_velocity * dt) % (2 * math.pi)
        self.rotation_matrix = self._compute_rotation_matrix()

    def get_world_transformed_nodes(self) -> List[Tuple[float, float, float, float]]:
        """Returns nodes rotated into camera space with current 3D orientation."""
        transformed = []
        for sx, sy, sz, sr in self.nodes:
            pos = self.rotation_matrix @ np.array([sx, sy, sz])
            transformed.append((pos[0], pos[1], pos[2], sr))
        return transformed

