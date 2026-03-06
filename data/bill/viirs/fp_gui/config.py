"""
viirs/fp_gui/config.py
Configuration and constants for the Fire Accumulation Viewer.
"""

# Default scatter size
DEFAULT_SCATTER_SIZE = 5

# Animation interval in milliseconds
DEFAULT_ANIMATION_INTERVAL_MS = 100

# Max parallel threads for shapefile I/O
MAX_WORKERS = 32

# ---- Colour levels ----
N_COLOUR_LEVELS = 100

# Colourmap endpoints (RGBA)
COLOUR_NEWEST = (1.0, 0.0, 0.0, 1.0)          # bright red
COLOUR_OLDEST = (0.55, 0.27, 0.07, 1.0)        # warm brown — visible on white

# Default popup columns
DEFAULT_POPUP_COLUMNS = [
    "latitude", "longitude", "utm_x", "utm_y",
    "FRP_MW", "confidence", "T4", "T5",
    "day", "line", "sample",
    "detection_datetime", "age_days",
]

# Filename regex
FILENAME_DATETIME_PATTERN = r"(\d{8}T\d{4})"
FILENAME_DATETIME_FORMAT = "%Y%m%dT%H%M"

# Raster display defaults
RASTER_ALPHA = 1
RASTER_CMAP = "gray"

# ---- Performance: raster downsampling ----
# Longest edge limit for the *display* copy of the raster.
# Full-res is kept; matplotlib only renders the downsampled copy.
# 2000 px is sharp on 1080p–1440p yet fast to pan/zoom.
MAX_RASTER_DISPLAY_DIM = 2000