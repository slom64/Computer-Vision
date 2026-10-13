"""
Fluid Bed Pellet & Agglomeration Simulation Package
===================================================
A high-fidelity simulation engine designed to generate photorealistic synthetic
training datasets for detecting single pellets and multi-pellet agglomerates (2, 3, 4+ pellets)
in pharmaceutical fluid bed imaging systems.
"""

from .config import SimulationConfig
from .pellet_models import PelletEntity, PelletType
from .optical_renderer import OpticalRenderer
from .background_engine import BackgroundEngine
from .particle_physics import FluidBedKinematics
from .exporter import DatasetExporter
from .temporal_verifier import TemporalAgglomerationVerifier

__all__ = [
    'SimulationConfig',
    'PelletEntity',
    'PelletType',
    'OpticalRenderer',
    'BackgroundEngine',
    'FluidBedKinematics',
    'DatasetExporter',
    'TemporalAgglomerationVerifier'
]

