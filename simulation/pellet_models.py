import math
import random
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np

class PelletType(Enum):
    SINGLE = "single"
    DIMER = "agglomerate_2"
    TRIMER = "agglomerate_3"
    CLUSTER = "agglomerate_cluster"
    MACRO = "agglomerate_macro"

class PelletEntity:
    """
    Represents an individual physical entity in 3D fluid-bed space:
    either a single pellet or an agglomerate of multiple connected spheres (2 to 20 pellets).
    Supports multiple agglomeration morphologies:
    - 'globular': Compact dense 3D raspberry cluster (common in heavy binder deposition)
    - 'chain': Serpentine linear or curved chain (capillary filament bridging)
    - 'branched': Dendritic branched network (multi-contact collisions)
    - 'random_tree': Natural stochastic adhesion
    """
    def __init__(
        self,
        entity_id: int,
        count: int,
        base_radius: float = 13.0,
        radius_variance: float = 0.14,
        overlap_range: Tuple[float, float] = (0.80, 0.92),
        morphology: str = 'mixed'
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
        elif count <= 6:
            self.pellet_type = PelletType.CLUSTER
        else:
            self.pellet_type = PelletType.MACRO
            
        # Select morphology
        if morphology == 'mixed' or morphology == 'auto':
            if count <= 3:
                self.morphology = 'chain'
            else:
                self.morphology = random.choice(['globular', 'globular', 'chain', 'branched', 'random_tree'])
        else:
            self.morphology = morphology
            
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
        """Generates 3D constituent spherical nodes for single or agglomerated pellets (1 to 20 pellets)."""
        nodes: List[Tuple[float, float, float, float]] = []
        r0 = self.base_radius * random.uniform(1.0 - radius_variance, 1.0 + radius_variance)
        nodes.append((0.0, 0.0, 0.0, r0))
        
        if self.count == 1:
            return nodes
            
        # Directional memory for chain growth
        chain_theta = random.uniform(0, 2 * math.pi)
        chain_phi = random.uniform(0.3 * math.pi, 0.7 * math.pi)
        
        for step in range(self.count - 1):
            r_child = self.base_radius * random.uniform(1.0 - radius_variance, 1.0 + radius_variance)
            overlap = random.uniform(overlap_range[0], overlap_range[1])
            
            # --- Case 1: Chain Morphology (Linear / Curved Filament) ---
            if self.morphology == 'chain':
                parent = nodes[-1]
                chain_theta += random.gauss(0, 0.45)
                chain_phi += random.gauss(0, 0.35)
                chain_phi = max(0.15 * math.pi, min(0.85 * math.pi, chain_phi))
                
                dist = (parent[3] + r_child) * overlap
                dx = dist * math.sin(chain_phi) * math.cos(chain_theta)
                dy = dist * math.sin(chain_phi) * math.sin(chain_theta)
                dz = dist * math.cos(chain_phi)
                nodes.append((parent[0] + dx, parent[1] + dy, parent[2] + dz, r_child))
                continue
                
            # --- Case 2: Globular Morphology (Dense Raspberry Cluster) ---
            if self.morphology == 'globular':
                # Attach close to center of mass
                com_x = sum(n[0] for n in nodes) / len(nodes)
                com_y = sum(n[1] for n in nodes) / len(nodes)
                com_z = sum(n[2] for n in nodes) / len(nodes)
                
                # Pick a surface parent closest to COM
                parent = min(nodes, key=lambda n: (n[0]-com_x)**2 + (n[1]-com_y)**2 + (n[2]-com_z)**2)
                
                best_candidate = None
                min_com_dist = float('inf')
                
                # Sample candidate attachment angles
                for _ in range(8):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = random.uniform(0.1 * math.pi, 0.9 * math.pi)
                    dist = (parent[3] + r_child) * overlap
                    cx = parent[0] + dist * math.sin(phi) * math.cos(theta)
                    cy = parent[1] + dist * math.sin(phi) * math.sin(theta)
                    cz = parent[2] + dist * math.cos(phi)
                    
                    # Ensure no inter-penetration with existing nodes (must be at least 1.4 * r away from all others)
                    collision = False
                    for existing in nodes:
                        c_dist = math.sqrt((cx - existing[0])**2 + (cy - existing[1])**2 + (cz - existing[2])**2)
                        if c_dist < (existing[3] + r_child) * 0.72:
                            collision = True
                            break
                    if not collision:
                        dist_to_com = (cx - com_x)**2 + (cy - com_y)**2 + (cz - com_z)**2
                        if dist_to_com < min_com_dist:
                            min_com_dist = dist_to_com
                            best_candidate = (cx, cy, cz, r_child)
                            
                if best_candidate is not None:
                    nodes.append(best_candidate)
                else:
                    # Fallback
                    theta = random.uniform(0, 2 * math.pi)
                    phi = random.uniform(0.15 * math.pi, 0.85 * math.pi)
                    dist = (parent[3] + r_child) * overlap
                    nodes.append((
                        parent[0] + dist * math.sin(phi) * math.cos(theta),
                        parent[1] + dist * math.sin(phi) * math.sin(theta),
                        parent[2] + dist * math.cos(phi),
                        r_child
                    ))
                continue
                
            # --- Case 3: Branched / Dendritic Morphology ---
            if self.morphology == 'branched':
                # Pick parent with low index or random to create branches
                parent = random.choice(nodes[:max(1, len(nodes) // 2)])
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0.15 * math.pi, 0.85 * math.pi)
                dist = (parent[3] + r_child) * overlap
                nodes.append((
                    parent[0] + dist * math.sin(phi) * math.cos(theta),
                    parent[1] + dist * math.sin(phi) * math.sin(theta),
                    parent[2] + dist * math.cos(phi),
                    r_child
                ))
                continue
                
            # --- Case 4: Random Tree Morphology ---
            parent = random.choice(nodes)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0.15 * math.pi, 0.85 * math.pi)
            dist = (parent[3] + r_child) * overlap
            nodes.append((
                parent[0] + dist * math.sin(phi) * math.cos(theta),
                parent[1] + dist * math.sin(phi) * math.sin(theta),
                parent[2] + dist * math.cos(phi),
                r_child
            ))
            
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
