#!/usr/bin/env python3
"""
viirs/fp_gui/__main__.py
VIIRS Fire Pixel Accumulation Viewer
=====================================
Entry point.  Run with:
    python main.py
Or import and use programmatically:
    from fire_gui import FireAccumulationGUI
    app = FireAccumulationGUI()
    app.run()
"""

import sys
import os

# Ensure the fp_gui package directory is on the path (for bare imports like `from config import ...`)
_fp_gui_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _fp_gui_dir)

# Ensure the parent of viirs/ is on the path (for `from viirs.utils.xxx import ...`)
_viirs_parent = os.path.dirname(os.path.dirname(_fp_gui_dir))
if _viirs_parent not in sys.path:
    sys.path.insert(0, _viirs_parent)

# Propagate to subprocesses (e.g. `python3 -m viirs.utils.shapify`)
_existing_pypath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    _fp_gui_dir + os.pathsep + _viirs_parent
    + (os.pathsep + _existing_pypath if _existing_pypath else "")
)

from fire_gui import FireAccumulationGUI


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="VIIRS Fire Pixel Accumulation Viewer")
    parser.add_argument(
        "raster", nargs="?", default=None,
        help="Optional path to a raster (.bin) file to load on startup")
    args = parser.parse_args()

    app = FireAccumulationGUI(raster_path=args.raster)
    app.run()


if __name__ == "__main__":
    main()