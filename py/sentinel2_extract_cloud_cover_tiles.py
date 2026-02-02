'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror '''

#!/usr/bin/env python3
"""
Plot cloud cover over time by tile ID from cached Sentinel-2 results.

Usage:
    python plot_cloud_cover.py [--cache-dir=DIR] [--output=FILE]

Options:
    --cache-dir=DIR   Cache directory (default: .sentinel2_cache)
    --output=FILE     Output plot filename (default: cloud_cover_by_tile.png)
    --show            Show interactive plot instead of saving
"""

import os
import sys
import pickle
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CACHE_DIR = ".sentinel2_cache"
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")


def extract_tile_id(product_id: str) -> str:
    """Extract tile ID from product filename."""
    parts = product_id.split("_")
    for p in parts:
        if p.startswith("T") and len(p) == 6:
            return p
    return "UNKNOWN"


def extract_date(product_id: str) -> datetime:
    """
    Extract sensing date from product filename.
    Format: S2A_MSIL2A_20240503T202851_N0510_R114_T09WWQ_20240504T013252.zip
    The date is in the third field: YYYYMMDDTHHMMSS
    """
    parts = product_id.split("_")
    for p in parts:
        # Look for YYYYMMDDTHHMMSS pattern
        if len(p) == 15 and p[8] == "T":
            try:
                return datetime.strptime(p[:8], "%Y%m%d")
            except ValueError:
                continue
    return None


def load_all_cached_results(cache_dir: str) -> List[Tuple[str, float]]:
    """Load all cached results from the results directory."""
    results_dir = os.path.join(cache_dir, "results")

    if not os.path.exists(results_dir):
        print(f"ERROR: Results cache directory not found: {results_dir}")
        sys.exit(1)

    results = []
    pkl_files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]

    print(f"Loading {len(pkl_files)} cached results...")

    for filename in pkl_files:
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                product_id, cloud_pct = pickle.load(f)
                results.append((product_id, cloud_pct))
        except Exception as e:
            print(f"  Warning: Failed to load {filename}: {e}")

    print(f"Loaded {len(results)} results")
    return results


def organize_by_tile_and_date(results: List[Tuple[str, float]]) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    Organize results by tile ID, with date and cloud percentage.
    Returns: {tile_id: [(date, cloud_pct), ...]}
    """
    by_tile = defaultdict(list)

    for product_id, cloud_pct in results:
        tile_id = extract_tile_id(product_id)
        sensing_date = extract_date(product_id)

        if sensing_date is None:
            print(f"  Warning: Could not extract date from {product_id}")
            continue

        by_tile[tile_id].append((sensing_date, cloud_pct))

    # Sort each tile's data by date
    for tile_id in by_tile:
        by_tile[tile_id].sort(key=lambda x: x[0])

    return dict(by_tile)


def aggregate_by_day(tile_data: List[Tuple[datetime, float]]) -> Tuple[List[datetime], List[float]]:
    """
    Aggregate multiple observations on the same day by averaging.
    Returns: (dates, cloud_percentages)
    """
    by_day = defaultdict(list)

    for dt, cloud_pct in tile_data:
        day = dt.date()
        by_day[day].append(cloud_pct)

    # Average multiple observations on same day
    days = sorted(by_day.keys())
    dates = [datetime.combine(d, datetime.min.time()) for d in days]
    averages = [sum(by_day[d]) / len(by_day[d]) for d in days]

    return dates, averages


def plot_cloud_cover(data_by_tile: Dict[str, List[Tuple[datetime, float]]],
                     output_file: str = None,
                     show: bool = False):
    """
    Create a line plot of cloud cover over time, one line per tile.
    """
    if not data_by_tile:
        print("ERROR: No data to plot")
        return

    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map for different tiles
    colors = plt.cm.tab20(range(len(data_by_tile)))

    # Plot each tile
    for i, (tile_id, tile_data) in enumerate(sorted(data_by_tile.items())):
        dates, cloud_pcts = aggregate_by_day(tile_data)

        ax.plot(dates, cloud_pcts,
                marker='o',
                markersize=3,
                linewidth=1,
                alpha=0.8,
                color=colors[i],
                label=f"{tile_id} ({len(tile_data)} obs)")

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cloud Cover (%)", fontsize=12)
    ax.set_title("Sentinel-2 Cloud Cover by Tile Over Time", fontsize=14)

    # Set y-axis limits
    ax.set_ylim(0, 100)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Grid
    ax.grid(True, alpha=0.3)

    # Legend - outside the plot if many tiles
    if len(data_by_tile) > 10:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, ncol=1)
    else:
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save or show
    if show:
        plt.show()
    else:
        if output_file is None:
            output_file = "cloud_cover_by_tile.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")

    plt.close()


def print_summary(data_by_tile: Dict[str, List[Tuple[datetime, float]]]):
    """Print summary statistics for each tile."""
    print("\nSummary by Tile:")
    print("-" * 70)
    print(f"{'Tile':<10} {'Obs':>6} {'Min':>8} {'Max':>8} {'Avg':>8} {'Date Range'}")
    print("-" * 70)

    for tile_id in sorted(data_by_tile.keys()):
        tile_data = data_by_tile[tile_id]
        cloud_values = [c for _, c in tile_data]
        dates = [d for d, _ in tile_data]

        min_cloud = min(cloud_values)
        max_cloud = max(cloud_values)
        avg_cloud = sum(cloud_values) / len(cloud_values)
        date_range = f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"

        print(f"{tile_id:<10} {len(tile_data):>6} {min_cloud:>7.2f}% {max_cloud:>7.2f}% {avg_cloud:>7.2f}% {date_range}")


def main():
    # Parse arguments
    cache_dir = CACHE_DIR
    output_file = "cloud_cover_by_tile.png"
    show = False

    for arg in sys.argv[1:]:
        if arg.startswith("--cache-dir="):
            cache_dir = arg.split("=")[1]
        elif arg.startswith("--output="):
            output_file = arg.split("=")[1]
        elif arg == "--show":
            show = True
        elif arg in ["-h", "--help"]:
            print(__doc__)
            sys.exit(0)

    print(f"Cache directory: {cache_dir}")

    # Load cached results
    results = load_all_cached_results(cache_dir)

    if not results:
        print("ERROR: No cached results found")
        sys.exit(1)

    # Organize by tile and date
    data_by_tile = organize_by_tile_and_date(results)

    print(f"Found {len(data_by_tile)} unique tiles")

    # Print summary
    print_summary(data_by_tile)

    # Create plot
    print(f"\nGenerating plot...")
    plot_cloud_cover(data_by_tile, output_file=output_file, show=show)


if __name__ == "__main__":
    main()
