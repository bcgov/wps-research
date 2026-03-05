"""
viirs/fp_gui/config.py

Configuration and constants for the Fire Accumulation Viewer.
"""

# Default scatter size
DEFAULT_SCATTER_SIZE = 10

# Animation interval in milliseconds
DEFAULT_ANIMATION_INTERVAL_MS = 500

# Max parallel threads for shapefile I/O
MAX_WORKERS = 64

# ---- Colour levels ----
# Number of discrete colour levels from newest (red) to oldest (pale yellow).
# Each day ages the pixel by one level.  Pixels older than N_COLOUR_LEVELS
# stay at the palest colour.
N_COLOUR_LEVELS = 100

# Colourmap endpoints (RGBA)
COLOUR_NEWEST = (1.0, 0.0, 0.0, 1.0)       # bright red
COLOUR_OLDEST = (1.0, 1.0, 0.6, 1.0)       # pale yellow

# Default popup columns to display when clicking a pixel
DEFAULT_POPUP_COLUMNS = [
    "latitude", "longitude", "utm_x", "utm_y",
    "FRP_MW", "confidence", "T4", "T5",
    "day", "line", "sample",
    "detection_datetime", "age_days",
]

# Filename regex pattern:  VIIRS_VNP14IMG_UTM<YYYYMMDD>T<HHMM>
FILENAME_DATETIME_PATTERN = r"(\d{8}T\d{4})"
FILENAME_DATETIME_FORMAT = "%Y%m%dT%H%M"

# Raster display defaults
RASTER_ALPHA = 0.8
RASTER_CMAP = "gray"