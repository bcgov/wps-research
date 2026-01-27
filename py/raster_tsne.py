#!/usr/bin/env python3
"""
20260127: ENVI Raster t-SNE Embedding Tool

This script reads an n-dimensional ENVI format raster (32-bit float, BSQ),
runs cuML t-SNE dimensionality reduction, and outputs an ENVI format file
with the embedding dimensions as bands.

Supports subsampling with nearest-neighbor interpolation for non-sampled pixels.
"""

import argparse
import numpy as np
from osgeo import gdal
import sys
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Suppress GDAL warnings for cleaner output
gdal.UseExceptions()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run cuML t-SNE on an ENVI format raster file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_file',
        help='Input ENVI raster file (without .hdr extension)'
    )
    parser.add_argument(
        'output_file',
        help='Output ENVI raster file (without .hdr extension)'
    )
    parser.add_argument(
        '-d', '--n_components',
        type=int,
        default=3,
        help='Number of embedding dimensions'
    )
    parser.add_argument(
        '-s', '--n_skip',
        type=int,
        default=1,
        help='Use every n-th pixel (1 = use all pixels)'
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='t-SNE perplexity parameter'
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1000,
        help='Number of t-SNE iterations'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=200.0,
        help='t-SNE learning rate'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--nodata',
        type=float,
        default=None,
        help='NoData value to exclude from processing'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark mode: iterate from n_skip=2048 down to n_skip=1, '
             'timing each run and plotting progress with exponential extrapolation'
    )
    parser.add_argument(
        '--benchmark_plot',
        type=str,
        default='benchmark_progress.png',
        help='Output path for benchmark plot (default: benchmark_progress.png)'
    )
    return parser.parse_args()


def read_envi_raster(input_file, verbose=False):
    """
    Read ENVI format raster using GDAL.
    
    Returns:
        data: numpy array of shape (bands, lines, samples)
        dataset: GDAL dataset for metadata extraction
    """
    if verbose:
        print(f"Opening input file: {input_file}")
    
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Failed to open {input_file}")
    
    n_bands = dataset.RasterCount
    n_cols = dataset.RasterXSize  # samples
    n_rows = dataset.RasterYSize  # lines
    
    if verbose:
        print(f"  Dimensions: {n_cols} samples x {n_rows} lines x {n_bands} bands")
        print(f"  Data type: {gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)}")
    
    # Read all bands into a 3D array (bands, rows, cols)
    data = np.zeros((n_bands, n_rows, n_cols), dtype=np.float32)
    for i in range(n_bands):
        band = dataset.GetRasterBand(i + 1)
        data[i] = band.ReadAsArray().astype(np.float32)
    
    return data, dataset


def create_sample_indices(n_rows, n_cols, n_skip):
    """
    Create indices for subsampled pixels.
    
    Returns:
        sample_rows: row indices for sampled pixels
        sample_cols: column indices for sampled pixels
    """
    row_indices = np.arange(0, n_rows, n_skip)
    col_indices = np.arange(0, n_cols, n_skip)
    
    # Create meshgrid of sample positions
    sample_cols, sample_rows = np.meshgrid(col_indices, row_indices)
    
    return sample_rows.ravel(), sample_cols.ravel()


def run_tsne(data, n_components, perplexity, n_iter, learning_rate, random_state, verbose=False):
    """
    Run cuML t-SNE on the input data.
    
    Args:
        data: 2D numpy array of shape (n_samples, n_features)
        n_components: number of embedding dimensions
        
    Returns:
        embedded: 2D numpy array of shape (n_samples, n_components)
    """
    try:
        from cuml.manifold import TSNE
        if verbose:
            print("Using cuML GPU-accelerated t-SNE")
    except ImportError:
        print("Warning: cuML not available, falling back to sklearn t-SNE (slower)")
        from sklearn.manifold import TSNE
    
    if verbose:
        print(f"Running t-SNE with {n_components} components...")
        print(f"  Input shape: {data.shape}")
        print(f"  Perplexity: {perplexity}")
        print(f"  Iterations: {n_iter}")
    
    # Adjust perplexity if necessary
    n_samples = data.shape[0]
    adjusted_perplexity = min(perplexity, (n_samples - 1) / 3)
    if adjusted_perplexity != perplexity:
        print(f"  Adjusted perplexity to {adjusted_perplexity:.1f} (sample size constraint)")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=1 if verbose else 0
    )
    
    embedded = tsne.fit_transform(data)
    
    if verbose:
        print(f"  Output shape: {embedded.shape}")
    
    return np.array(embedded, dtype=np.float32)


def interpolate_missing_pixels(data, sample_rows, sample_cols, embedded, 
                                n_rows, n_cols, n_components, nodata_mask, verbose=False):
    """
    Interpolate embedding values for non-sampled pixels using nearest neighbor.
    
    For each non-sampled pixel, find the nearest sampled pixel in feature space
    and use its embedding value.
    """
    if verbose:
        print("Interpolating non-sampled pixels using nearest neighbor in feature space...")
    
    n_bands = data.shape[0]
    
    # Create output array
    output = np.zeros((n_components, n_rows, n_cols), dtype=np.float32)
    
    # Fill in sampled pixels
    for i, (r, c) in enumerate(zip(sample_rows, sample_cols)):
        if not nodata_mask[r, c]:
            output[:, r, c] = embedded[i]
    
    # Get feature vectors for sampled pixels (excluding nodata)
    valid_sample_mask = ~nodata_mask[sample_rows, sample_cols]
    valid_sample_indices = np.where(valid_sample_mask)[0]
    
    if len(valid_sample_indices) == 0:
        print("Warning: No valid sampled pixels found!")
        return output
    
    valid_sample_rows = sample_rows[valid_sample_indices]
    valid_sample_cols = sample_cols[valid_sample_indices]
    valid_embedded = embedded[valid_sample_indices]
    
    # Get feature vectors for valid sampled pixels
    sample_features = data[:, valid_sample_rows, valid_sample_cols].T  # (n_valid_samples, n_bands)
    
    # Try to use cuML for GPU-accelerated nearest neighbor
    try:
        from cuml.neighbors import NearestNeighbors
        import cupy as cp
        use_gpu = True
        if verbose:
            print("  Using cuML GPU-accelerated nearest neighbor search")
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        use_gpu = False
        if verbose:
            print("  Using sklearn nearest neighbor search (slower)")
    
    # Fit nearest neighbor model on sampled features
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
    nn.fit(sample_features)
    
    # Process non-sampled pixels in chunks to manage memory
    chunk_size = 100000
    
    # Create mask of pixels that need interpolation
    all_rows, all_cols = np.mgrid[0:n_rows, 0:n_cols]
    all_rows = all_rows.ravel()
    all_cols = all_cols.ravel()
    
    # Create set of sampled pixel linear indices for fast lookup
    sampled_linear = set(sample_rows * n_cols + sample_cols)
    
    # Find non-sampled pixels
    non_sampled_mask = np.array([
        (r * n_cols + c) not in sampled_linear and not nodata_mask[r, c]
        for r, c in zip(all_rows, all_cols)
    ])
    
    non_sampled_rows = all_rows[non_sampled_mask]
    non_sampled_cols = all_cols[non_sampled_mask]
    
    n_to_interpolate = len(non_sampled_rows)
    if verbose:
        print(f"  Interpolating {n_to_interpolate} pixels...")
    
    # Process in chunks
    for start in range(0, n_to_interpolate, chunk_size):
        end = min(start + chunk_size, n_to_interpolate)
        
        chunk_rows = non_sampled_rows[start:end]
        chunk_cols = non_sampled_cols[start:end]
        
        # Get feature vectors for this chunk
        chunk_features = data[:, chunk_rows, chunk_cols].T
        
        # Find nearest neighbors
        _, indices = nn.kneighbors(chunk_features)
        indices = np.array(indices).ravel()
        
        # Assign embedding values from nearest sampled pixel
        for i, (r, c) in enumerate(zip(chunk_rows, chunk_cols)):
            output[:, r, c] = valid_embedded[indices[i]]
        
        if verbose and (end - start) == chunk_size:
            print(f"    Processed {end}/{n_to_interpolate} pixels...")
    
    return output


def write_envi_raster(output_file, data, src_dataset, verbose=False):
    """
    Write output ENVI format raster with proper georeferencing.
    
    Args:
        output_file: output file path
        data: numpy array of shape (n_components, n_rows, n_cols)
        src_dataset: source GDAL dataset for georeferencing
    """
    n_bands, n_rows, n_cols = data.shape
    
    if verbose:
        print(f"Writing output file: {output_file}")
        print(f"  Dimensions: {n_cols} samples x {n_rows} lines x {n_bands} bands")
    
    # Create ENVI format driver
    driver = gdal.GetDriverByName('ENVI')
    
    # Create output dataset (Float32)
    out_dataset = driver.Create(
        output_file,
        n_cols,
        n_rows,
        n_bands,
        gdal.GDT_Float32
    )
    
    if out_dataset is None:
        raise RuntimeError(f"Failed to create output file: {output_file}")
    
    # Copy georeferencing from source
    out_dataset.SetGeoTransform(src_dataset.GetGeoTransform())
    out_dataset.SetProjection(src_dataset.GetProjection())
    
    # Write bands
    for i in range(n_bands):
        band = out_dataset.GetRasterBand(i + 1)
        band.WriteArray(data[i])
        band.SetDescription(f"t-SNE dimension {i + 1}")
    
    # Flush and close
    out_dataset.FlushCache()
    out_dataset = None
    
    if verbose:
        print("  Output file written successfully")


def exponential_model(x, a, b, c):
    """Exponential model: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def power_model(x, a, b):
    """Power model: y = a * (n_samples)^b where n_samples ~ 4^(-log2_skip)"""
    # x is log2(n_skip), so n_samples proportional to 4^(-x)
    # y = a * 4^(-b*x) = a * exp(-b * x * ln(4))
    return a * np.exp(-b * x * np.log(4))


def update_benchmark_plot(log2_skips, times, plot_path, total_pixels):
    """
    Update the benchmark plot with current data and exponential extrapolation.
    
    Args:
        log2_skips: list of log2(n_skip) values (descending order in time)
        times: list of elapsed times in seconds
        plot_path: path to save the plot
        total_pixels: total number of pixels in the image
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert to numpy arrays
    x_data = np.array(log2_skips)
    y_data = np.array(times)
    
    # Plot actual measurements
    ax.scatter(x_data, y_data, s=100, c='blue', zorder=5, label='Measured times')
    ax.plot(x_data, y_data, 'b-', linewidth=2, alpha=0.7)
    
    # Annotate points with timing info
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        n_skip = 2 ** int(x)
        n_samples = total_pixels // (n_skip * n_skip)
        label = f'{y:.1f}s\n({n_samples:,} px)'
        ax.annotate(label, (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=8)
    
    estimated_time_at_1 = None
    
    # Fit exponential model if we have enough points
    if len(x_data) >= 3:
        try:
            # Use power model: time ~ a * (n_samples)^b
            # Since n_samples ~ total_pixels / (n_skip^2) = total_pixels * 4^(-log2_skip)
            # We fit: time = a * exp(-b * log2_skip * ln(4))
            
            # Initial guess
            p0 = [times[-1], 1.5]  # a = latest time, b = ~1.5 (between linear and quadratic)
            
            popt, pcov = curve_fit(power_model, x_data, y_data, p0=p0, maxfev=10000)
            a_fit, b_fit = popt
            
            # Generate smooth curve for plotting
            x_smooth = np.linspace(min(x_data), 0, 100)
            y_smooth = power_model(x_smooth, *popt)
            
            ax.plot(x_smooth, y_smooth, 'r--', linewidth=2, 
                    label=f'Power fit: t = {a_fit:.2e} × n_samples^{b_fit:.2f}')
            
            # Estimate time at log2(n_skip) = 0 (i.e., n_skip = 1)
            estimated_time_at_1 = power_model(0, *popt)
            
            # Plot the extrapolated point
            ax.scatter([0], [estimated_time_at_1], s=150, c='red', marker='*', 
                       zorder=6, label=f'Estimated n_skip=1: {format_time(estimated_time_at_1)}')
            
            # Add vertical line at n_skip=1
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            
        except Exception as e:
            print(f"  Warning: Could not fit exponential model: {e}")
    
    # Labels and formatting
    ax.set_xlabel('log₂(n_skip)', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('t-SNE Benchmark: Computation Time vs. Subsampling Rate', fontsize=14)
    
    # Create custom x-tick labels showing both log2 and actual n_skip
    x_ticks = list(range(int(max(x_data)), -1, -1))
    x_tick_labels = [f'{x}\n(n_skip={2**x})' for x in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    
    # Set y-axis to log scale if range is large
    if len(y_data) > 1 and max(y_data) / min(y_data) > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add info box
    info_text = f'Total pixels: {total_pixels:,}\nRuns completed: {len(times)}'
    if estimated_time_at_1 is not None:
        info_text += f'\nEstimated full run: {format_time(estimated_time_at_1)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return estimated_time_at_1


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def run_computation(data, sample_rows, sample_cols, nodata_mask, n_rows, n_cols,
                    n_components, perplexity, n_iter, learning_rate, random_state, 
                    n_skip, verbose=False):
    """
    Run the core computation (t-SNE + interpolation) and return results.
    
    Returns:
        output: the embedded output array
        elapsed: computation time in seconds
    """
    start_time = time.time()
    
    n_samples = len(sample_rows)
    n_bands = data.shape[0]
    
    # Extract feature vectors for sampled pixels
    sample_data = data[:, sample_rows, sample_cols].T  # (n_samples, n_bands)
    
    # Handle nodata in sampled pixels
    sample_nodata_mask = nodata_mask[sample_rows, sample_cols]
    
    # Replace nodata with zeros for t-SNE (they'll be masked in output)
    sample_data_clean = sample_data.copy()
    sample_data_clean[sample_nodata_mask] = 0
    
    # Also replace any remaining NaN/Inf values
    sample_data_clean = np.nan_to_num(sample_data_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Run t-SNE
    embedded = run_tsne(
        sample_data_clean,
        n_components,
        perplexity,
        n_iter,
        learning_rate,
        random_state,
        verbose
    )
    
    # Create output array
    if n_skip == 1:
        # All pixels were sampled, just reshape
        if verbose:
            print("Reshaping output (no interpolation needed)...")
        
        output = np.zeros((n_components, n_rows, n_cols), dtype=np.float32)
        for i, (r, c) in enumerate(zip(sample_rows, sample_cols)):
            output[:, r, c] = embedded[i]
        
        # Set nodata pixels to NaN
        output[:, nodata_mask] = np.nan
    else:
        # Interpolate non-sampled pixels
        output = interpolate_missing_pixels(
            data, sample_rows, sample_cols, embedded,
            n_rows, n_cols, n_components, nodata_mask, verbose
        )
        
        # Set nodata pixels to NaN
        output[:, nodata_mask] = np.nan
    
    elapsed = time.time() - start_time
    return output, elapsed


def run_benchmark(args, data, src_dataset, nodata_mask, n_rows, n_cols):
    """
    Run benchmark mode: iterate from n_skip=2048 down, timing each run.
    """
    print("=" * 60)
    print("BENCHMARK MODE")
    print("=" * 60)
    
    total_pixels = n_rows * n_cols
    print(f"Image dimensions: {n_cols} x {n_rows} = {total_pixels:,} pixels")
    print(f"Starting benchmark from n_skip=2048, halving each iteration")
    print("=" * 60)
    
    # Track results
    log2_skips = []
    times = []
    last_output = None
    
    # Start from n_skip = 2048 and halve each time
    n_skip = 2048
    
    while n_skip >= 1:
        log2_skip = int(np.log2(n_skip))
        
        # Create sample indices for this n_skip
        sample_rows, sample_cols = create_sample_indices(n_rows, n_cols, n_skip)
        n_samples = len(sample_rows)
        
        # Skip if too few samples for t-SNE
        valid_samples = ~nodata_mask[sample_rows, sample_cols]
        n_valid = np.sum(valid_samples)
        
        min_samples = max(10, int(args.perplexity * 3) + 1)
        if n_valid < min_samples:
            print(f"\nn_skip={n_skip} (log2={log2_skip}): Skipping - only {n_valid} valid samples "
                  f"(need at least {min_samples} for perplexity={args.perplexity})")
            n_skip //= 2
            continue
        
        print(f"\n{'='*60}")
        print(f"Running n_skip={n_skip} (log2={log2_skip})")
        print(f"  Sampling {n_samples:,} pixels ({100*n_samples/total_pixels:.2f}%)")
        print(f"  Valid samples: {n_valid:,}")
        
        # Run computation with timing
        output, elapsed = run_computation(
            data, sample_rows, sample_cols, nodata_mask, n_rows, n_cols,
            args.n_components, args.perplexity, args.n_iter, args.learning_rate,
            args.random_state, n_skip, verbose=args.verbose
        )
        
        last_output = output
        
        # Record results
        log2_skips.append(log2_skip)
        times.append(elapsed)
        
        print(f"  Computation time: {format_time(elapsed)}")
        
        # Update plot
        estimated = update_benchmark_plot(log2_skips, times, args.benchmark_plot, total_pixels)
        print(f"  Plot updated: {args.benchmark_plot}")
        
        if estimated is not None:
            print(f"  Estimated time for n_skip=1: {format_time(estimated)}")
        
        # Halve n_skip for next iteration
        n_skip //= 2
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    # Print summary table
    print("\nSummary:")
    print(f"{'n_skip':>8} {'log2':>6} {'Samples':>12} {'Time':>12}")
    print("-" * 42)
    for log2_skip, t in zip(log2_skips, times):
        n_skip_val = 2 ** log2_skip
        n_samples = total_pixels // (n_skip_val * n_skip_val)
        print(f"{n_skip_val:>8} {log2_skip:>6} {n_samples:>12,} {format_time(t):>12}")
    
    return last_output


def main():
    args = parse_args()
    
    if args.verbose or args.benchmark:
        print("=" * 60)
        print("ENVI Raster t-SNE Embedding Tool")
        print("=" * 60)
    
    # Read input raster
    data, src_dataset = read_envi_raster(args.input_file, args.verbose or args.benchmark)
    n_bands, n_rows, n_cols = data.shape
    
    # Create nodata mask
    if args.nodata is not None:
        # Pixel is nodata if ANY band has nodata value
        nodata_mask = np.any(data == args.nodata, axis=0)
        if args.verbose or args.benchmark:
            nodata_count = np.sum(nodata_mask)
            print(f"NoData pixels: {nodata_count} ({100*nodata_count/(n_rows*n_cols):.1f}%)")
    else:
        # Check for NaN values
        nodata_mask = np.any(np.isnan(data), axis=0)
        if (args.verbose or args.benchmark) and np.any(nodata_mask):
            nodata_count = np.sum(nodata_mask)
            print(f"NaN pixels detected: {nodata_count}")
    
    # Benchmark mode
    if args.benchmark:
        output = run_benchmark(args, data, src_dataset, nodata_mask, n_rows, n_cols)
        
        if output is not None:
            # Write the last completed output
            write_envi_raster(args.output_file, output, src_dataset, verbose=True)
        
        # Clean up
        src_dataset = None
        
        print("\n" + "=" * 60)
        print("Benchmark complete!")
        print(f"Final plot saved to: {args.benchmark_plot}")
        print(f"Output file (last completed): {args.output_file}")
        print("=" * 60)
        
        return 0
    
    # Normal mode (non-benchmark)
    # Create sample indices
    sample_rows, sample_cols = create_sample_indices(n_rows, n_cols, args.n_skip)
    n_samples = len(sample_rows)
    
    if args.verbose:
        total_pixels = n_rows * n_cols
        print(f"Sampling: {n_samples} of {total_pixels} pixels "
              f"({100*n_samples/total_pixels:.1f}%) with n_skip={args.n_skip}")
    
    # Extract feature vectors for sampled pixels
    sample_data = data[:, sample_rows, sample_cols].T  # (n_samples, n_bands)
    
    # Handle nodata in sampled pixels
    sample_nodata_mask = nodata_mask[sample_rows, sample_cols]
    valid_samples = ~sample_nodata_mask
    
    if args.verbose:
        valid_count = np.sum(valid_samples)
        print(f"Valid sampled pixels: {valid_count} of {n_samples}")
    
    # Replace nodata with zeros for t-SNE (they'll be masked in output)
    sample_data_clean = sample_data.copy()
    sample_data_clean[sample_nodata_mask] = 0
    
    # Also replace any remaining NaN/Inf values
    sample_data_clean = np.nan_to_num(sample_data_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Run t-SNE
    embedded = run_tsne(
        sample_data_clean,
        args.n_components,
        args.perplexity,
        args.n_iter,
        args.learning_rate,
        args.random_state,
        args.verbose
    )
    
    # Create output array
    if args.n_skip == 1:
        # All pixels were sampled, just reshape
        if args.verbose:
            print("Reshaping output (no interpolation needed)...")
        
        output = np.zeros((args.n_components, n_rows, n_cols), dtype=np.float32)
        for i, (r, c) in enumerate(zip(sample_rows, sample_cols)):
            output[:, r, c] = embedded[i]
        
        # Set nodata pixels to NaN
        output[:, nodata_mask] = np.nan
    else:
        # Interpolate non-sampled pixels
        output = interpolate_missing_pixels(
            data, sample_rows, sample_cols, embedded,
            n_rows, n_cols, args.n_components, nodata_mask, args.verbose
        )
        
        # Set nodata pixels to NaN
        output[:, nodata_mask] = np.nan
    
    # Write output
    write_envi_raster(args.output_file, output, src_dataset, args.verbose)
    
    # Clean up
    src_dataset = None
    
    if args.verbose:
        print("=" * 60)
        print("Done!")
        print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
