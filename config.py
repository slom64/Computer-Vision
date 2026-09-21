WIDTH = 1200
HEIGHT = 800

NUM_SINGLE_PELLETS = 150
NUM_AGGLOMERATES = 40
MAX_PELLETS_PER_AGGLOMERATE = 6

BASE_RADIUS = 0.1 # Meters
BOX_SIZE = 10.0 # Much larger box to spread them out

# Camera view limits (the "visible" bounding area in world space)
CAM_X_MIN = -BOX_SIZE / 2.0
CAM_X_MAX = BOX_SIZE / 2.0
CAM_Y_MIN = -BOX_SIZE / 2.0
CAM_Y_MAX = BOX_SIZE / 2.0
CAM_Z_MIN = 0.0
CAM_Z_MAX = BOX_SIZE

# AI Pipeline Constants
SPAWN_Z = CAM_Z_MAX + 5.0    # Spawn safely above the perspective frustum
DESPAWN_Z = CAM_Z_MIN - 5.0  # Despawn safely below the perspective frustum
NDC_MARGIN = 0.05            # 5% margin on the screen edges for visibility check

# ML Dataset Generation Options
RECORD_FPS = 10.0            # Frames per second to save to dataset
IMG_WIDTH = 512              # Exported Image Width
IMG_HEIGHT = 512             # Exported Image Height

# Physics Simulation Options
DT = 1.0 / 60.0
RESTITUTION = 0.8