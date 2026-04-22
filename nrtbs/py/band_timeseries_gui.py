'''
A GUI which allows you to visualize the band values inside a square as a time series
>>> interactive_time_serise('bin_fil_dir', 'image', 10)
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as mwidgets
from dnbr import NBR
from operator import add, sub
import datetime
import numpy as np
import math
from misc import extract_date
import os
import argparse
from concurrent.futures import ProcessPoolExecutor

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(15,8))
clicks = []
plot_colors = ['b','r','y','k','c','m']

_load_cache = {}  # file_dir -> (sorted_file_names, params) — avoids re-loading across calls
current_image_index = 0
current_plot_type = 'image'

# Navigation button globals (kept alive to avoid garbage collection)
_btn_prev = None
_btn_next = None


def _load_all(file_dir):
    '''Load all bin files in file_dir. Cached per-directory for the Python session.'''
    if file_dir in _load_cache:
        print(f'using cached data for {file_dir} ({len(_load_cache[file_dir][0])} files)')
        return _load_cache[file_dir]
    files = os.listdir(file_dir)
    file_list = [f for f in files if f.endswith('.bin')]
    sorted_file_names = sorted(file_list, key=extract_date)
    paths = [f'{file_dir}/{f}' for f in sorted_file_names]
    n_workers = min(16, max(1, (os.cpu_count() or 4) // 4))
    print(f'loading {len(paths)} files with {n_workers} workers (first time for this dir)')
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        params_ = list(ex.map(NBR, paths))
    _load_cache[file_dir] = (sorted_file_names, params_)
    return sorted_file_names, params_


def _draw_reference():
    '''Redraw the reference image (ax1) using current_image_index and current_plot_type.
    Preserves click squares.'''
    ax1.clear()
    ref = params[current_image_index]
    if current_plot_type == 'image':
        image = np.stack([scale(ref[0]), scale(ref[1]), scale(ref[2])], axis=2)
        ax1.imshow(image)
    elif current_plot_type == 'nbr':
        ax1.imshow(ref[4], cmap='grey')
    ax1.set_title(f'[{current_image_index % len(filenames)}] {filenames[current_image_index]}  ({current_plot_type})')
    for i, (cx, cy) in enumerate(clicks):
        color = plot_colors[i % len(plot_colors)]
        ax1.add_patch(patches.Rectangle((cx, cy), square_width, square_width,
                                        linewidth=1, edgecolor=color, facecolor='none'))
    fig.canvas.draw()


def _go_prev(event=None):
    '''Navigate to the previous image in the sequence.'''
    global current_image_index
    current_image_index = (current_image_index - 1) % len(filenames)
    print(f'reference -> [{current_image_index}] {filenames[current_image_index]}')
    _draw_reference()


def _go_next(event=None):
    '''Navigate to the next image in the sequence.'''
    global current_image_index
    current_image_index = (current_image_index + 1) % len(filenames)
    print(f'reference -> [{current_image_index}] {filenames[current_image_index]}')
    _draw_reference()


def interactive_time_serise(file_dir, plot_type: str, width, image_index=0):
    '''
    Interactive time-series GUI for a directory of 4-band bin files.

    Parameters
    ----------
    file_dir : str
        Path to the directory containing .bin files to load. Each file is
        expected to be readable by dnbr.NBR and contain at least 5 bands
        (B12, B11, B09, B08, NBR). Defaults to the current working directory
        ('.') when called from the CLI without --dir.
    plot_type : str
        Initial reference-image display mode. Must be one of:
          'image' — false-colour RGB composite (bands 0, 1, 2 scaled to [0,1])
          'nbr'   — greyscale Normalized Burn Ratio (band 4)
        Any other value silently falls back to 'image'.
    width : int
        Side length (in pixels) of the sampling square drawn on each click.
        The mean and std of each band are computed over this square for every
        date in the time series.
    image_index : int, optional
        Index (into the date-sorted file list) of the file to show as the
        initial reference image.  0 = earliest (default), -1 = most recent.
        Negative indices are supported (Python-style).

    Controls once the window is open
    ---------------------------------
      left-click on the image : add a sampling square + time-series traces
      right-click              : clear all squares and traces
      left / right arrow       : switch reference image to previous / next date
      t                        : toggle between 'image' and 'nbr' reference views
      c                        : clear squares/traces (same as right-click)
      Back / Next buttons      : on-screen buttons to navigate images

    The bin files for a given directory are cached in memory after the first
    load, so re-calling this function with the same directory is instant.
    '''
    sorted_file_names, params_ = _load_all(file_dir)

    global params, filenames, square_width, current_image_index, current_plot_type
    global _btn_prev, _btn_next
    params = params_
    filenames = sorted_file_names
    square_width = width
    current_image_index = image_index if image_index >= 0 else len(filenames) + image_index
    current_image_index = max(0, min(current_image_index, len(filenames) - 1))
    current_plot_type = plot_type if plot_type in ('image', 'nbr') else 'image'

    print('available files (sorted by date):')
    for i, name in enumerate(sorted_file_names):
        marker = ' <- reference' if i == current_image_index else ''
        print(f'  [{i}] {name}{marker}')
    print("controls: left-click=add square, right-click=clear, ←/→=change date, t=toggle image/nbr")

    clicks.clear()
    for a in (ax2, ax3, ax4, ax5, ax6):
        a.clear()

    # Make room at the bottom for navigation buttons
    fig.subplots_adjust(bottom=0.10)

    ax_prev = fig.add_axes([0.35, 0.01, 0.1, 0.04])
    ax_next = fig.add_axes([0.55, 0.01, 0.1, 0.04])
    _btn_prev = mwidgets.Button(ax_prev, '← Back')
    _btn_next = mwidgets.Button(ax_next, 'Next →')
    _btn_prev.on_clicked(_go_prev)
    _btn_next.on_clicked(_go_next)

    _draw_reference()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def on_key(event):
    global current_image_index, current_plot_type
    if event.key in ('right', 'n'):
        _go_next()
    elif event.key in ('left', 'p'):
        _go_prev()
    elif event.key == 't':
        current_plot_type = 'nbr' if current_plot_type == 'image' else 'image'
        print(f'plot_type -> {current_plot_type}')
        _draw_reference()
    elif event.key == 'c':
        _clear_all()


def _clear_all():
    clicks.clear()
    for p in list(ax1.patches):
        p.remove()
    for a in (ax2, ax3, ax4, ax5, ax6):
        a.clear()
    fig.canvas.draw()
    print('cleared all squares and traces')

    
def param_plots(clicks, width):
    '''
    takes the list of click locations and square_width and plots the B12, B11, B09, B08, and NBR of the mean value timeseries in a box with side length = square_width
    '''
    ax = [ax2, ax3, ax4, ax5, ax6]
    band_names = ['B12', 'B11', 'B09', 'B08', 'NBR']
    n_bands = len(band_names)
    mean = [[] for _ in range(n_bands)]
    std = [[] for _ in range(n_bands)]

    time = []
    y = int(clicks[-1][1])
    x = int(clicks[-1][0])

    for file in range(len(params)):
        date = datetime.datetime.strptime(filenames[file].split('_')[2].split('T')[0], '%Y%m%d')
        for n in range(n_bands):
            patch = params[file][n][y:y+width, x:x+width]
            mean[n].append(np.nanmean(patch))
            std[n].append(np.nanstd(patch))
        time.append(date)

    color = plot_colors[(len(clicks) - 1) % len(plot_colors)]
    for band in range(n_bands):
        ax[band].plot(time, mean[band], color=color, label=f'Mean at ({x},{y})')
        ax[band].plot(time, list(map(add, mean[band], std[band])), color=color, linestyle='dashed')
        ax[band].plot(time, list(map(sub, mean[band], std[band])), color=color, linestyle='dotted')
        ax[band].legend()
        ax[band].set_title(band_names[band])

    plt.tight_layout()
    fig.canvas.draw()


def on_click(event):
    '''
    Left-click on the reference image to add a sampling square and time-series traces.
    Right-click (anywhere) to clear all squares and traces.
    Clicks beyond the color palette size wrap around to the start of the palette.
    '''
    if event.button == 3:
        _clear_all()
        return

    if event.inaxes is not ax1:
        return

    clicks.append((event.xdata, event.ydata))
    color = plot_colors[(len(clicks) - 1) % len(plot_colors)]
    print(f'click #{len(clicks)} at ({event.xdata:.1f}, {event.ydata:.1f}) color={color}')

    square = patches.Rectangle((event.xdata, event.ydata), square_width, square_width,
                               linewidth=1, edgecolor=color, facecolor='none')
    ax1.add_patch(square)
    fig.canvas.draw()
    param_plots(clicks, square_width)

        
def scale(X):
    '''
    Used to scale the image if necesary
    '''
    # default: scale a band to [0, 1]  and then clip
    mymin = np.nanmin(X) # np.nanmin(X))
    mymax = np.nanmax(X) # np.nanmax(X))
    X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    X[X < 0.] = 0.  # clip
    X[X > 1.] = 1.

    # use histogram trimming / turn it off to see what this step does!
    if  True:
        values = X.ravel().tolist()
        values.sort()
        n_pct = 1. # percent for stretch value
        frac = n_pct / 100.
        lower = int(math.floor(float(len(values))*frac))
        upper = int(math.floor(float(len(values))*(1. - frac)))
        mymin, mymax = values[lower], values[upper]
        X = (X-mymin) / (mymax - mymin)  # perform the linear transformation
    
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive time-series viewer for multi-band .bin satellite imagery.',
        epilog=('Controls: left-click=add square, right-click=clear, '
                '←/→=change date, t=toggle image/nbr, c=clear, '
                'or use the on-screen Back/Next buttons.')
    )
    parser.add_argument(
        '--dir', '-d',
        default='.',
        help=('Path to the directory containing .bin files. '
              'Each file must be readable by dnbr.NBR. '
              'Default: current working directory.')
    )
    parser.add_argument(
        '--plot-type', '-p',
        choices=['image', 'nbr'],
        default='image',
        help=("Initial reference display mode. "
              "'image' = false-colour RGB composite, "
              "'nbr' = greyscale Normalized Burn Ratio. "
              "Default: 'image'.")
    )
    parser.add_argument(
        '--width', '-w',
        type=int,
        default=10,
        help=('Side length in pixels of the sampling square drawn on '
              'each click. Default: 10.')
    )
    parser.add_argument(
        '--index', '-i',
        type=int,
        default=0,
        help=('Index into the date-sorted file list for the initial '
              'reference image. 0 = earliest (default), -1 = most recent. '
              'Negative indices are supported.')
    )
    args = parser.parse_args()
    interactive_time_serise(args.dir, args.plot_type, args.width, args.index)
