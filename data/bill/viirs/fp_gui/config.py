"""
viirs/fp_gui/config.py
Configuration and constants for the Fire Accumulation Viewer.
"""

# Default scatter size (base multiplier — actual pixel size scales with zoom)
# When a raster is loaded this is auto-computed from VNP14IMG_PIXEL_SIZE_M
# divided by the raster's spatial resolution.
DEFAULT_SCATTER_SIZE = 1

# Animation interval in milliseconds
DEFAULT_ANIMATION_INTERVAL_MS = 100

# Max parallel threads for shapefile I/O
MAX_WORKERS = 16

# ---- Colour levels ----
N_COLOUR_LEVELS = 100

# Colourmap endpoints (RGBA)
COLOUR_NEWEST = (1.0, 0.0, 0.0, 1.0)          # bright red
COLOUR_OLDEST = (0.55, 0.27, 0.07, 1.0)        # warm brown — visible on white

# Default popup columns
DEFAULT_POPUP_COLUMNS = [
    "latitude", "longitude", "x", "y",
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
# Set to 99999 so NO downsampling by default.
# Lower in Config dialog to speed up rendering on large rasters.
MAX_RASTER_DISPLAY_DIM = 99999

# ---- Performance: pan preview downsampling ----
# Max pixel dimension for the low-res raster preview shown during panning.
# Lower = faster pan, coarser preview.  Only affects raster, not fire pixels.
PAN_PREVIEW_MAX_DIM = 800

# ---- VNP14IMG fire pixel ground size (metres) ----
VNP14IMG_PIXEL_SIZE_M = 375