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

# Ensure the package directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fire_gui import FireAccumulationGUI


def main():
    app = FireAccumulationGUI()
    app.run()


if __name__ == "__main__":
    main()