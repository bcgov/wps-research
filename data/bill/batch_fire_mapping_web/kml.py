"""KML export for accepted fires.

Called from ``_accept_fire_sync`` after the canonical output dir is
populated. Polygonizes ``<FIRE>_crop.bin_classified.bin`` to a shapefile
and reprojects to EPSG:4326 KML so the result opens cleanly in Google
Earth.

Failures are warned to stderr and swallowed: KML is a deliverable, not
part of the analytical truth, so a missing ogr2ogr binary or an empty
classification must not abort the accept.
"""

import os
import shutil
import subprocess
import sys


_POLYGONIZE = os.environ.get(
    'WPS_BINARY_POLYGONIZE',
    '/home/bill/GitHub/wps-research/py/binary_polygonize.py')


def _export_kml(fire_numbe: str, fire_dir: str) -> str | None:
    """Generate ``<FIRE>.kml`` (EPSG:4326) from the accepted classification.

    Returns the KML path on success, ``None`` on any failure (with a
    warning printed to stderr). Never raises.
    """
    clf_bin = os.path.join(
        fire_dir, f'{fire_numbe}_crop.bin_classified.bin')
    if not os.path.isfile(clf_bin):
        sys.stderr.write(
            f'[kml] WARNING: {fire_numbe}: classified raster missing at '
            f'{clf_bin}; skipping KML export.\n')
        sys.stderr.flush()
        return None

    if not shutil.which('ogr2ogr'):
        sys.stderr.write(
            f'[kml] WARNING: {fire_numbe}: ogr2ogr not on PATH; '
            f'skipping KML export.\n')
        sys.stderr.flush()
        return None

    if not os.path.isfile(_POLYGONIZE):
        sys.stderr.write(
            f'[kml] WARNING: {fire_numbe}: polygonize script missing at '
            f'{_POLYGONIZE}; skipping KML export.\n')
        sys.stderr.flush()
        return None

    raw_shp = clf_bin + '.shp'
    final_kml = os.path.join(fire_dir, f'{fire_numbe}.kml')

    # binary_polygonize.py exits with status 1 even on success (legacy
    # quirk — see py/binary_polygonize.py:145). Don't check the return
    # code; verify success by looking for the .shp on disk. Capture
    # stderr so a real crash gets surfaced rather than masked by the
    # always-1 exit code.
    poly_stderr = b''
    try:
        proc = subprocess.run(
            [sys.executable, _POLYGONIZE, clf_bin],
            cwd=fire_dir,
            capture_output=True,
            timeout=300,
        )
        poly_stderr = proc.stderr or b''
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(
            f'[kml] WARNING: {fire_numbe}: polygonize failed: {exc}\n')
        sys.stderr.flush()
        return None

    if not os.path.isfile(raw_shp):
        tail = poly_stderr.decode(errors='replace').strip().splitlines()[-3:]
        detail = ' | '.join(tail) if tail else '(no stderr)'
        sys.stderr.write(
            f'[kml] WARNING: {fire_numbe}: shapefile not produced '
            f'(empty classification or polygonize crash); '
            f'last stderr: {detail}\n')
        sys.stderr.flush()
        return None

    # Drop the wrong-CRS KML polygonize emits as a side effect — we
    # produce the proper EPSG:4326 one below.
    stale_kml = clf_bin + '.kml'
    if os.path.isfile(stale_kml):
        try:
            os.remove(stale_kml)
        except OSError:
            pass

    # Rename the shapefile sidecar set from the verbose
    # ``<FIRE>_crop.bin_classified.bin.{ext}`` to a clean
    # ``<FIRE>.{ext}``. Done before ogr2ogr so the KML's <Document>
    # name reflects the friendly stem.
    shp_to_use = raw_shp
    renamed_any = False
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        src = clf_bin + ext
        dst = os.path.join(fire_dir, f'{fire_numbe}{ext}')
        if not os.path.isfile(src):
            continue
        try:
            os.replace(src, dst)
            renamed_any = True
            if ext == '.shp':
                shp_to_use = dst
        except OSError as exc:
            sys.stderr.write(
                f'[kml] WARNING: {fire_numbe}: rename {src} -> {dst}: '
                f'{exc}\n')
            sys.stderr.flush()
    if renamed_any and shp_to_use == raw_shp:
        # .shp itself failed to rename — fall back to the raw path so
        # ogr2ogr still has something to read.
        shp_to_use = raw_shp

    try:
        result = subprocess.run(
            ['ogr2ogr', '-f', 'KML', '-t_srs', 'EPSG:4326',
             '-overwrite', final_kml, shp_to_use],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(
            f'[kml] WARNING: {fire_numbe}: ogr2ogr failed: {exc}\n')
        sys.stderr.flush()
        return None

    if result.returncode != 0 or not os.path.isfile(final_kml):
        sys.stderr.write(
            f'[kml] WARNING: {fire_numbe}: ogr2ogr returned '
            f'{result.returncode}; '
            f'stderr: {result.stderr.decode(errors="replace").strip()}\n')
        sys.stderr.flush()
        return None

    return final_kml
