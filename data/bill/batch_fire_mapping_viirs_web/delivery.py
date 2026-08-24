"""Naming and manifest for the delivered product archive.

Two jobs:

1. Work out the acquisition datetime the delivered products should be
   named for, and render the agreed pattern
   ``YY_<fire>_YYYYMMDD_HHMM_detection_sentinel2``.

2. Describe every file in the archive, so the recipient does not have
   to infer what ``_kgc_selected.bin`` is from its name.

The datetime is the newest Sentinel-2 acquisition that contributed to
the imagery the classification was run on, converted from UTC to
Vancouver local time. How confidently that is known differs by source,
and the difference is recorded rather than hidden -- see
``acquisition_datetime``.
"""

import os
import re
import sys
import glob
import json
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:                                   # pragma: no cover
    ZoneInfo = None

# Always resolved through the tz database rather than a fixed -7/-8, so
# a historical fire converts with the offset that was in force then, and
# any future rule change is picked up with the system tz data.
LOCAL_TZ = 'America/Vancouver'


def _to_local(dt_utc: str):
    """'YYYYMMDDTHHMMSS' (UTC) -> aware datetime in Vancouver time."""
    txt = (dt_utc or '').strip().replace('-', '').replace(':', '')
    m = re.match(r'^(\d{8})T?(\d{6})$', txt)
    if not m:
        m2 = re.match(r'^(\d{8})$', txt)
        if not m2:
            return None
        txt = m2.group(1) + 'T000000'
        m = re.match(r'^(\d{8})T?(\d{6})$', txt)
    try:
        naive = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
    except ValueError:
        return None
    aware = naive.replace(tzinfo=timezone.utc)
    if ZoneInfo is None:
        return aware
    try:
        return aware.astimezone(ZoneInfo(LOCAL_TZ))
    except Exception as exc:
        sys.stderr.write(f'[delivery] timezone lookup failed: {exc}\n')
        return aware


def acquisition_datetime(fire, stack_path: str = '') -> dict:
    """Newest acquisition datetime behind this fire's imagery.

    Returns ``{'utc', 'local', 'source', 'exact'}``; ``exact`` says
    whether the answer is known or estimated.

    L2 recent: EXACT. The application composited the product itself and
    recorded, per contributing acquisition, the datetime from the source
    file name.

    MRAP: ESTIMATED. That compositing happens in a back-end process
    which does not report which pixels it took, so the best available
    answer is the newest acquisition the back end would have considered
    for the tiles covering this AOI. It may name an acquisition that
    contributed nothing here.
    """
    src = (getattr(fire, 'post_source', '') or 'l2').lower()

    if src == 'l2':
        side = ''
        try:
            from .l2_recent import date_polygons_path
            if stack_path:
                side = date_polygons_path(stack_path)
            elif getattr(fire, 'crop_bin', ''):
                side = date_polygons_path(fire.crop_bin)
        except Exception:
            side = ''
        if side and os.path.isfile(side):
            try:
                with open(side, encoding='utf-8') as fh:
                    newest = (json.load(fh).get('acq_newest_utc') or '')
                if newest:
                    return {'utc': newest, 'local': _to_local(newest),
                            'source': 'l2', 'exact': True}
            except Exception as exc:
                sys.stderr.write(
                    f'[delivery] could not read {side}: {exc}\n')
        # Older products predate the datetime being recorded; fall
        # through to the estimate rather than failing the download.

    return _estimate_from_tiles(fire, exact=False, src=src)


def _estimate_from_tiles(fire, exact: bool, src: str) -> dict:
    """Newest acquisition available over this AOI's tiles."""
    try:
        from .l2_recent import (tiles_intersecting_bbox, zips_for_tile,
                                L2RecentError)
        from osgeo import gdal
        ref = ''
        try:
            from . import state as _st
        except Exception:
            _st = None
        if _st is not None:
            ref = (_st.state.rasters_by_year.get(fire.fire_year)
                   or _st.state.raster_path or '')
        proj = ''
        if ref:
            ds = gdal.Open(ref, gdal.GA_ReadOnly)
            if ds is not None:
                proj = ds.GetProjection()
                ds = None
        tiles = tiles_intersecting_bbox(fire.bbox_native, proj)
        newest = ''
        cap = (getattr(fire, 'l2_start_date', '') or '')
        for t in tiles:
            for _key, acq8, _tok, path in zips_for_tile(t):
                if cap and acq8 > cap:
                    continue
                parts = os.path.basename(path).split('_')
                dt = parts[2] if len(parts) > 2 else ''
                if dt and dt > newest:
                    newest = dt
        if newest:
            return {'utc': newest, 'local': _to_local(newest),
                    'source': src, 'exact': exact}
    except Exception as exc:
        sys.stderr.write(f'[delivery] tile scan failed: {exc}\n')
    return {'utc': '', 'local': None, 'source': src, 'exact': False}


def delivery_stem(fire_numbe: str, acq: dict) -> str:
    """``YY_<fire>_YYYYMMDD_HHMM_detection_sentinel2``.

    The fire number is used as given. Fires created before a BCWS number
    was issued carry whatever name the operator chose, which is the
    documented behaviour rather than a defect.
    """
    local = (acq or {}).get('local')
    if local is None:
        return ''
    return (f'{local:%y}_{fire_numbe}_{local:%Y%m%d}_{local:%H%M}'
            f'_detection_sentinel2')


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

# Descriptions are keyed by a recognisable part of the file name. The
# first match wins, so more specific keys are listed first.
_DESCRIPTIONS = [
    ('_detection_sentinel2.kml',
     'Fire perimeter as KML, for Google Earth and web mapping.'),
    ('_detection_sentinel2.shp',
     'Fire perimeter as an ESRI shapefile (geometry).'),
    ('_detection_sentinel2.shx', 'Shapefile geometry index.'),
    ('_detection_sentinel2.dbf', 'Shapefile attribute table.'),
    ('_detection_sentinel2.prj',
     'Shapefile coordinate reference system definition.'),
    ('_detection_sentinel2.cpg', 'Shapefile character encoding.'),
    ('_detection_sentinel2.bin',
     'Accepted fire classification raster (ENVI): 1 = burned, '
     '0 = not burned, on the AOI grid.'),
    ('_kgc_selected.bin',
     'Diagnostic: the clusters the algorithm selected as burned, '
     'before the brushing cleanup pass.'),
    ('_classified.bin',
     'Fire classification raster as delivered by the accepted run '
     '(same content as the dated .bin above).'),
    ('_params.yaml',
     'Every parameter used for the accepted run: algorithm settings, '
     'brush settings, band selection and imagery source.'),
    ('_perimeter.kml', 'Fire perimeter as KML (legacy name).'),
    ('_perimeter.shp', 'Fire perimeter as a shapefile (legacy name).'),
    ('_perimeter.shx', 'Shapefile geometry index (legacy name).'),
    ('_perimeter.dbf', 'Shapefile attribute table (legacy name).'),
    ('_perimeter.prj', 'Shapefile CRS definition (legacy name).'),
    ('_perimeter.cpg', 'Shapefile character encoding (legacy name).'),
    ('MANIFEST', 'This file: what every file in the archive is.'),
]

_PREVIEW_DESCRIPTIONS = [
    ('geo.json',
     'Georeferencing for the images in this folder: for each view, the '
     'geotransform and raster size, so a preview can be placed on a '
     'map or compared pixel-for-pixel with the rasters above.'),
    ('result_prebrush',
     'ML classification before the brushing cleanup pass, drawn over '
     'the post-fire imagery.'),
    ('result',
     'ML classification (the accepted perimeter) drawn over the '
     'post-fire imagery.'),
    ('brush_comparison',
     'Before/after comparison of the brushing cleanup pass.'),
    ('comparison', 'Pre-fire and post-fire imagery side by side.'),
    ('hint', 'The hint layer: the independent burned-area estimate '
             'used by the clustering and by the agreement score.'),
    ('post', 'Post-fire imagery composite for the AOI.'),
    ('pre', 'Pre-fire imagery composite for the AOI.'),
    ('diff', 'Difference between pre-fire and post-fire imagery, '
             'highlighting change.'),
]


def describe(rel_path: str) -> str:
    """One line saying what a file in the archive is."""
    name = os.path.basename(rel_path)
    if name.lower().endswith(('.hdr', '.aux.xml')):
        # Headers travel with their raster and are not described
        # separately.
        return ''
    if rel_path.startswith('previews' + os.sep) or \
            rel_path.startswith('previews/'):
        for key, text in _PREVIEW_DESCRIPTIONS:
            if key in name:
                return text
        return 'Rendered view of the AOI.'
    for key, text in _DESCRIPTIONS:
        if key in name:
            return text
    return ''


def build_manifest_pdf(out_path: str, fire_numbe: str, files: list,
                       acq: dict, stem: str) -> bool:
    """Write the archive contents listing. False if it could not."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, \
            ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                        Spacer, Table, TableStyle)
    except Exception as exc:
        sys.stderr.write(f'[delivery] reportlab unavailable: {exc}\n')
        return False

    ss = getSampleStyleSheet()
    navy = colors.HexColor('#1F3864')
    h = ParagraphStyle('h', parent=ss['Title'], fontSize=15,
                       textColor=navy, spaceAfter=4)
    sub = ParagraphStyle('s', parent=ss['Normal'], fontSize=9.5,
                         alignment=1,
                         textColor=colors.HexColor('#555555'),
                         spaceAfter=12)
    body = ParagraphStyle('b', parent=ss['Normal'], fontSize=9,
                          leading=11.5, spaceAfter=6)
    cell = ParagraphStyle('c', parent=ss['Normal'], fontSize=8,
                          leading=10)
    hdr = ParagraphStyle('hc', parent=cell, textColor=colors.white,
                         fontName='Helvetica-Bold')

    story = [Paragraph('ARCHIVE CONTENTS', h),
             Paragraph(f'Fire mapping products &ndash; {fire_numbe}', sub)]

    local = (acq or {}).get('local')
    if local is not None:
        when = local.strftime('%Y-%m-%d %H:%M %Z')
        exact = (acq or {}).get('exact')
        src = 'L2 recent' if (acq or {}).get('source') == 'l2' else 'MRAP'
        story.append(Paragraph(
            f'<b>Imagery source:</b> {src}<br/>'
            f'<b>Newest contributing acquisition:</b> {when} '
            f'(local) &nbsp;|&nbsp; {acq.get("utc")} UTC<br/>'
            f'<b>Product naming:</b> '
            f'<font face="Courier">{stem}.*</font>', body))
        if not exact:
            story.append(Paragraph(
                '<i>The acquisition time is an estimate. The MRAP '
                'composite is assembled by a back-end process that does '
                'not report which acquisitions supplied which pixels, '
                'so this is the newest acquisition available over the '
                'tiles covering this area rather than one confirmed to '
                'have contributed.</i>', body))
        else:
            story.append(Paragraph(
                '<i>The acquisition time is exact: this composite was '
                'assembled by the application, which recorded the '
                'datetime of every contributing acquisition.</i>', body))
    story.append(Spacer(1, 6))

    rows = [[Paragraph('File', hdr), Paragraph('What it is', hdr)]]
    for rel in files:
        d = describe(rel)
        if not d:
            continue
        rows.append([Paragraph(
            f'<font face="Courier" size="7.5">{rel}</font>', cell),
            Paragraph(d, cell)])
    t = Table(rows, colWidths=[2.75 * inch, 3.85 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9DB0CE')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#EEF2F8')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        'Files ending in <font face="Courier">.hdr</font> are ENVI '
        'headers describing the raster beside them (dimensions, data '
        'type, and the map projection) and are required to open it.',
        body))
    try:
        SimpleDocTemplate(
            out_path, pagesize=letter,
            leftMargin=0.85 * inch, rightMargin=0.85 * inch,
            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
            title=f'{fire_numbe} archive contents').build(story)
        return True
    except Exception as exc:
        sys.stderr.write(f'[delivery] manifest build failed: {exc}\n')
        return False
