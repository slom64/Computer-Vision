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
from .pellet_tracker import PelletTracker, KalmanBoxTracker
from .sequence_model import PelletSequenceModel
from .reflection_filter import MetalReflectionFilter
from .wall_sticking_monitor import WallStickingMonitor, AdhesionState

__all__ = [
    'SimulationConfig',
    'PelletEntity',
    'PelletType',
    'OpticalRenderer',
    'BackgroundEngine',
    'FluidBedKinematics',
    'DatasetExporter',
    'TemporalAgglomerationVerifier',
    'PelletTracker',
    'KalmanBoxTracker',
    'PelletSequenceModel',
    'MetalReflectionFilter',
    'WallStickingMonitor',
    'AdhesionState'
]


