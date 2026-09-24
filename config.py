# Simulation Window Display Dimensions
WIDTH = 1200
HEIGHT = 800

# Pellet Counts in Simulation
NUM_SINGLE_PELLETS = 140
NUM_AGGLOMERATES = 40
MAX_PELLETS_PER_AGGLOMERATE = 6

# Pellet Dimensions & Physics Space
# BASE_RADIUS 0.18m at camera distance ~10-12m projects to ~25-35 pixels at 1024x1024,
# which precisely matches real industrial fluid-bed imaging sensors.
BASE_RADIUS = 0.18  # Meters
BOX_SIZE = 10.0     # Simulation bounding volume (Meters)

# Camera View Limits (Visible bounding area in world space)
CAM_X_MIN = -BOX_SIZE / 2.0
CAM_X_MAX = BOX_SIZE / 2.0
CAM_Y_MIN = -BOX_SIZE / 2.0
CAM_Y_MAX = BOX_SIZE / 2.0
CAM_Z_MIN = 0.0
CAM_Z_MAX = BOX_SIZE

# Spawn / Despawn Thresholds (Outside the camera perspective frustum)
SPAWN_Z = CAM_Z_MAX + 5.0
DESPAWN_Z = CAM_Z_MIN - 5.0
NDC_MARGIN = 0.05    # 5% margin inside the screen borders for CSV visibility filtering

# High-Resolution ML Dataset Export Options
RECORD_FPS = 10.0    # Simulation data capture rate (records per simulated second)
IMG_WIDTH = 1024     # High-Resolution Output Width
IMG_HEIGHT = 1024    # High-Resolution Output Height

# Deep Sequence Classification Parameters
SEQUENCE_LENGTH = 10 # Number of consecutive tumbling frames analyzed
CROP_SIZE = 96       # High-detail bounding box crop size for the ResNet-BiLSTM model

# Physics Engine Step Options
DT = 1.0 / 60.0
RESTITUTION = 0.8