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


def acquisition_datetime(fire, stack_path: str = '',
                         ref_raster: str = '') -> dict:
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

    return _estimate_from_tiles(fire, exact=False, src=src,
                                ref_raster=ref_raster)


def _estimate_from_tiles(fire, exact: bool, src: str,
                         ref_raster: str = '') -> dict:
    """Newest acquisition available over this AOI's tiles.

    The reference raster is supplied by the caller. Reaching for the
    application state from here did not work -- there is no module-level
    state object to import -- so this silently returned nothing, which
    is why delivered files kept their original names.
    """
    try:
        from .l2_recent import tiles_intersecting_bbox, zips_for_tile
        from osgeo import gdal
        ref = ref_raster or ''
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

    # Last resort: the date the loaded product was built for, which is
    # the leading YYYYMMDD of the stack file name. The time of day is
    # unknown, so it is reported as 00:00 and flagged inexact -- a dated
    # product name with an approximate time is far more use than every
    # file keeping its undated name.
    try:
        stem_name = os.path.basename(getattr(fire, 'crop_bin', '') or '')
        m = re.match(r'^(\d{8})_', stem_name)
        if not m:
            m = re.match(r'^(\d{8})$',
                         (getattr(fire, 'post_date', '') or '')[:8])
        if m:
            guess = m.group(1) + 'T000000'
            sys.stderr.write(
                f'[delivery] falling back to the product date '
                f'{m.group(1)} (time unknown)\n')
            return {'utc': guess, 'local': _to_local(guess),
                    'source': src, 'exact': False}
    except Exception as exc:
        sys.stderr.write(f'[delivery] date fallback failed: {exc}\n')

    return {'utc': '', 'local': None, 'source': src, 'exact': False}


def fallback_datetime(fire) -> dict:
    """Last resort so products are ALWAYS named to the convention.

    Uses the post date the stack was built for -- a real acquisition
    date, without the time of day. The time is reported as 0000 and the
    manifest states that the time was unavailable, so no precision is
    implied that is not there.
    """
    cand = ''
    m = re.search(r'(\d{8})_stack_',
                  os.path.basename(getattr(fire, 'crop_bin', '') or ''))
    if m:
        cand = m.group(1)
    if not cand:
        cand = str(getattr(fire, 'post_date', '') or '').replace('-', '')
    if not re.fullmatch(r'\d{8}', cand or ''):
        return {'utc': '', 'local': None, 'source': '', 'exact': False}
    return {'utc': cand + 'T000000',
            'local': _to_local(cand + 'T000000'),
            'source': (getattr(fire, 'post_source', '') or ''),
            'exact': False, 'time_unknown': True}


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

_IMAGERY_DESCRIPTIONS = [
    ('l2_most_recent',
     'Reflectance stack for the most recent L2 composite: the imagery '
     'the classification was run on, analysis-ready.'),
    ('l2_', 'Reflectance stack for the L2 composite built from this '
            'start date.'),
    ('mrap', 'Reflectance stack for the MRAP cloud-free composite.'),
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
    if rel_path.startswith('imagery/'):
        for key, text in _IMAGERY_DESCRIPTIONS:
            if key in rel_path:
                return text
        return 'Reflectance imagery stack for the AOI.'
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


def _normalise_preview_sizes(prev_dir: str, stage_dir: str) -> dict:
    """Same-size copies of any previews that differ from the rest.

    Returns {original_path: replacement_path}; originals are NEVER
    modified.

    Resizing in place was a genuine bug, not just untidiness: the
    download cache is keyed by a signature over these files, so
    rewriting them changed the signature that the archive had just been
    built for. The button went ready, the next poll saw a different
    signature, and it greyed itself again -- once per build, for ever.

    The views come from one raster and should already agree; one render
    path can disagree, and images of different sizes cannot be flicked
    through or stacked, so the outliers are matched to the majority.
    """
    out = {}
    if not os.path.isdir(prev_dir):
        return out
    try:
        from PIL import Image
    except Exception:
        return out
    sizes, imgs = {}, {}
    for fn in sorted(os.listdir(prev_dir)):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        if '.low.' in fn:
            continue
        full = os.path.join(prev_dir, fn)
        try:
            with Image.open(full) as im:
                imgs[full] = im.size
        except Exception:
            continue
        sizes[imgs[full]] = sizes.get(imgs[full], 0) + 1
    if len(sizes) <= 1:
        return out
    # Most common size wins; ties break toward the larger image so
    # detail is not thrown away wholesale.
    target = sorted(sizes.items(),
                    key=lambda kv: (kv[1], kv[0][0]))[-1][0]
    try:
        os.makedirs(stage_dir, exist_ok=True)
    except OSError:
        return out
    for full, size in imgs.items():
        if size == target:
            continue
        dst = os.path.join(stage_dir, os.path.basename(full))
        try:
            with Image.open(full) as im:
                im.resize(target, Image.LANCZOS).save(dst)
            out[full] = dst
            sys.stderr.write(
                f'[delivery] {os.path.basename(full)}: {size[0]}x'
                f'{size[1]} -> {target[0]}x{target[1]} in the archive '
                f'(original untouched)\n')
        except Exception as exc:
            sys.stderr.write(f'[delivery] resize failed for '
                             f'{full}: {exc}\n')
    return out


def build_archive(result_dir: str, fire_numbe: str, acq: dict,
                  out_zip: str, log=None, fire=None,
                  imagery=None) -> dict:
    """Zip an accepted fire directory as a delivered product set.

    Entry-by-entry rather than zipping the folder, so the delivered
    products can carry the dated naming pattern and a contents listing
    can be added without changing anything on disk.

    Returns a summary so the caller can log what actually happened
    instead of assuming it worked.
    """
    import zipfile

    stem = delivery_stem(fire_numbe, acq)
    if not stem:
        # Never ship un-renamed products just because the acquisition
        # time could not be established.
        acq = fallback_datetime(fire) if fire is not None else acq
        stem = delivery_stem(fire_numbe, acq)
        if stem:
            sys.stderr.write(
                '[delivery] acquisition time unavailable; naming from '
                'the stack post date with time 0000\n')
    VEC = ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.kml')
    fl = fire_numbe.lower()

    def arcname(rel: str) -> str:
        if not stem or os.path.dirname(rel):
            return rel                     # previews/ keep their names
        low = os.path.basename(rel).lower()
        ext = os.path.splitext(low)[1]
        if ext in VEC and ('_perimeter' in low or '_detection_' in low):
            return f'{stem}{ext}'
        if low == f'{fl}_classified.bin':
            return f'{stem}.bin'
        if low == f'{fl}_classified.hdr':
            return f'{stem}.hdr'
        return rel

    # Imagery stacks join THIS archive.
    #
    # They used to come from a separate "Download imagery" button: two
    # archives, two clicks, and nothing stating how they related.
    # De-duplicated by basename so nothing ships twice.
    extra, seen_names = [], set()
    for pth in (imagery or []):
        b = os.path.basename(pth)
        if b in seen_names or not os.path.isfile(pth):
            continue
        seen_names.add(b)
        extra.append(pth)

    # Uniform-size copies of any odd-sized previews, staged beside the
    # zip so the accepted directory is left exactly as it was.
    stage = os.path.join(os.path.dirname(out_zip) or '.',
                         f'.{fire_numbe}.previews.{os.getpid()}')
    swaps = _normalise_preview_sizes(
        os.path.join(result_dir, 'previews'), stage)

    entries = []
    for root, _dirs, fnames in os.walk(result_dir):
        for fn in sorted(fnames):
            if fn.startswith('.'):
                continue
            # Low-resolution progressive variants are a loading aid, not
            # a product. Filtered HERE as well as at accept time, so an
            # archive is clean even when the accepted directory was
            # populated before that filter existed.
            if '.low.' in fn:
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, result_dir).replace(os.sep, '/')
            entries.append((swaps.get(full, full), arcname(rel)))

    for pth in extra:
        b = os.path.basename(pth)
        m = re.search(r'_l2_d(\d{8})\.', b)
        if m:
            folder = f'imagery/l2_{m.group(1)}'
        elif '_l2.' in b:
            folder = 'imagery/l2_most_recent'
        else:
            folder = 'imagery/mrap'
        arc = f'{folder}/{b}'
        if not any(a == arc for _f, a in entries):
            entries.append((pth, arc))

    manifest_name = f'{fire_numbe}_ARCHIVE_CONTENTS.pdf'
    tmp_manifest = os.path.join(
        os.path.dirname(out_zip) or '.',
        f'.{fire_numbe}.manifest.{os.getpid()}.pdf')
    listed = sorted(a for _f, a in entries) + [manifest_name]
    have_manifest = False
    try:
        have_manifest = build_manifest_pdf(
            tmp_manifest, fire_numbe, listed, acq, stem or fire_numbe)
    except Exception as exc:
        sys.stderr.write(f'[delivery] manifest failed: {exc}\n')

    if not have_manifest:
        # reportlab is not installed everywhere; a PDF was asked for, so
        # write one directly rather than degrading to plain text.
        try:
            have_manifest = build_manifest_pdf_simple(
                tmp_manifest, fire_numbe, listed, acq,
                stem or fire_numbe)
            if have_manifest:
                sys.stderr.write(
                    '[delivery] manifest written without reportlab\n')
        except Exception as exc:
            sys.stderr.write(
                f'[delivery] built-in pdf writer failed: {exc}\n')

    if not have_manifest:
        # reportlab is not installed here. Write the PDF ourselves
        # rather than degrading the deliverable to a text file: the
        # archive goes to people outside the team, and "PDF if a
        # library happens to be present" is not a contract.
        have_manifest = _simple_pdf(
            tmp_manifest, f'{fire_numbe} archive contents',
            _manifest_lines(fire_numbe, listed, acq, stem or fire_numbe))
        if have_manifest:
            sys.stderr.write(
                '[delivery] manifest written without reportlab\n')

    if not have_manifest:
        # Both PDF paths failed: a text listing is better than nothing.
        manifest_name = f'{fire_numbe}_ARCHIVE_CONTENTS.txt'
        tmp_manifest = tmp_manifest[:-4] + '.txt'
        try:
            with open(tmp_manifest, 'w', encoding='utf-8') as fh:
                fh.write(f'ARCHIVE CONTENTS - {fire_numbe}\n\n')
                loc = (acq or {}).get('local')
                if loc is not None:
                    fh.write(
                        f'Newest contributing acquisition: '
                        f'{loc:%Y-%m-%d %H:%M %Z} (local), '
                        f'{(acq or {}).get("utc")} UTC\n')
                    fh.write(f'Exact: {bool((acq or {}).get("exact"))}\n')
                    fh.write(f'Product naming: {stem}.*\n\n')
                for rel in listed:
                    d = describe(rel)
                    if d:
                        fh.write(f'{rel}\n    {d}\n')
                fh.write('\nFiles ending in .hdr are ENVI headers for '
                         'the raster beside them.\n')
            have_manifest = True
        except Exception as exc:
            sys.stderr.write(f'[delivery] text manifest failed: {exc}\n')

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for full, arc in entries:
            zf.write(full, f'{fire_numbe}/{arc}')
        if have_manifest:
            zf.write(tmp_manifest, f'{fire_numbe}/{manifest_name}')
    try:
        if os.path.isfile(tmp_manifest):
            os.remove(tmp_manifest)
    except OSError:
        pass
    try:
        import shutil as _sh
        if os.path.isdir(stage):
            _sh.rmtree(stage, ignore_errors=True)
    except Exception:
        pass

    renamed = sum(1 for _f, a in entries
                  if stem and os.path.basename(a).startswith(stem))
    summary = {'stem': stem, 'entries': len(entries),
               'renamed': renamed, 'manifest': manifest_name
               if have_manifest else ''}
    msg = (f'[download] {fire_numbe}: {len(entries)} file(s), '
           f'{renamed} renamed to "{stem or "(unchanged)"}", '
           f'manifest={summary["manifest"] or "NONE"}')
    sys.stderr.write(msg + '\n')
    if log:
        log('  ' + msg.split('] ', 1)[1])
    return summary


# ---------------------------------------------------------------------
# Minimal PDF writer
# ---------------------------------------------------------------------
#
# reportlab is not installed everywhere this server runs, and the PDF
# manifest was silently degrading to a .txt as a result. A contents
# listing is plain text in a box, so it does not justify a dependency:
# this writes the few hundred bytes of PDF structure directly, using the
# built-in Helvetica/Courier fonts that every reader has.

def _pdf_escape(txt: str) -> str:
    return (str(txt).replace('\\', r'\\')
            .replace('(', r'\(').replace(')', r'\)'))


def _wrap(txt: str, width: int) -> list:
    words, lines, cur = str(txt).split(), [], ''
    for w in words:
        trial = f'{cur} {w}'.strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or ['']


def write_simple_pdf(out_path: str, lines: list) -> bool:
    """Write `lines` as a PDF. Each line is (text, font_key, size).

    font_key: 'B' bold, 'C' courier, anything else regular.
    """
    PAGE_W, PAGE_H = 612, 792
    LEFT, TOP, BOTTOM = 56, 736, 56

    pages, cur, y = [], [], TOP
    for text, font, size in lines:
        lead = size + 3.5
        if y - lead < BOTTOM:
            pages.append(cur)
            cur, y = [], TOP
        cur.append((text, font, size, y))
        y -= lead
    pages.append(cur)

    FONTS = {'B': '/F2', 'C': '/F3'}
    objs, streams = [], []
    for page in pages:
        parts = ['BT']
        for text, font, size, y in page:
            parts.append(f'{FONTS.get(font, "/F1")} {size} Tf')
            parts.append(f'1 0 0 1 {LEFT} {y:.1f} Tm')
            parts.append(f'({_pdf_escape(text)}) Tj')
        parts.append('ET')
        streams.append('\n'.join(parts).encode('latin-1', 'replace'))

    n_pages = len(streams)
    # 1 catalog, 2 pages, 3..(2+n) page objs, then contents, then fonts
    first_content = 3 + n_pages
    font_base = first_content + n_pages
    kids = ' '.join(f'{3 + i} 0 R' for i in range(n_pages))

    objs.append(b'<< /Type /Catalog /Pages 2 0 R >>')
    objs.append(f'<< /Type /Pages /Count {n_pages} /Kids [{kids}] >>'
                .encode())
    for i in range(n_pages):
        objs.append(
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} '
            f'{PAGE_H}] /Resources << /Font << /F1 {font_base} 0 R '
            f'/F2 {font_base + 1} 0 R /F3 {font_base + 2} 0 R >> >> '
            f'/Contents {first_content + i} 0 R >>'.encode())
    for st in streams:
        objs.append(b'<< /Length ' + str(len(st)).encode() + b' >>\n'
                    b'stream\n' + st + b'\nendstream')
    for base in ('Helvetica', 'Helvetica-Bold', 'Courier'):
        objs.append(f'<< /Type /Font /Subtype /Type1 /BaseFont '
                    f'/{base} /Encoding /WinAnsiEncoding >>'.encode())

    try:
        out = bytearray(b'%PDF-1.4\n')
        offsets = [0]
        for i, body in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f'{i} 0 obj\n'.encode() + body + b'\nendobj\n'
        xref = len(out)
        out += f'xref\n0 {len(objs) + 1}\n'.encode()
        out += b'0000000000 65535 f \n'
        for off in offsets[1:]:
            out += f'{off:010d} 00000 n \n'.encode()
        out += (f'trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n'
                f'startxref\n{xref}\n%%EOF\n').encode()
        with open(out_path, 'wb') as fh:
            fh.write(bytes(out))
        return True
    except Exception as exc:
        sys.stderr.write(f'[delivery] simple pdf failed: {exc}\n')
        return False


def build_manifest_pdf_simple(out_path: str, fire_numbe: str,
                              files: list, acq: dict, stem: str) -> bool:
    """The manifest, without reportlab."""
    L = []
    L.append(('ARCHIVE CONTENTS', 'B', 16))
    L.append((f'Fire mapping products - {fire_numbe}', '', 11))
    L.append(('', '', 6))
    loc = (acq or {}).get('local')
    if loc is not None:
        src = 'L2 recent' if (acq or {}).get('source') == 'l2' else 'MRAP'
        L.append((f'Imagery source: {src}', '', 9.5))
        L.append((f'Newest contributing acquisition: '
                  f'{loc:%Y-%m-%d %H:%M %Z} (local) / '
                  f'{(acq or {}).get("utc")} UTC', '', 9.5))
        L.append((f'Product naming: {stem}.*', 'C', 9))
        L.append(('', '', 4))
        if (acq or {}).get('exact'):
            note = ('The acquisition time is exact: this composite was '
                    'assembled by the application, which recorded the '
                    'datetime of every contributing acquisition.')
        else:
            note = ('The acquisition time is an ESTIMATE. It is the '
                    'newest acquisition available over the tiles '
                    'covering this area, not one confirmed to have '
                    'contributed pixels here.')
        for ln in _wrap(note, 92):
            L.append((ln, '', 8.5))
    L.append(('', '', 8))
    L.append(('FILES', 'B', 11))
    L.append(('', '', 4))
    for rel in files:
        d = describe(rel)
        if not d:
            continue
        L.append((rel, 'C', 8.5))
        for ln in _wrap(d, 96):
            L.append(('    ' + ln, '', 8.5))
        L.append(('', '', 2))
    L.append(('', '', 6))
    for ln in _wrap('Files ending in .hdr are ENVI headers describing '
                    'the raster beside them (dimensions, data type and '
                    'map projection) and are required to open it.', 92):
        L.append((ln, '', 8.5))
    return write_simple_pdf(out_path, L)


# ---------------------------------------------------------------------
# Minimal PDF writer
# ---------------------------------------------------------------------
# Used when reportlab is not installed on the server. The archive is a
# deliverable to people outside the team, so "PDF unless a library
# happens to be present" is not good enough -- this writes a valid PDF
# with nothing but the standard library.

def _pdf_escape(t: str) -> str:
    return (t.replace('\\', r'\\').replace('(', r'\(')
            .replace(')', r'\)'))


def _simple_pdf(out_path: str, title: str, lines: list) -> bool:
    """Write a plain multi-page PDF. `lines` are (style, text) pairs."""
    try:
        PW, PH = 612, 792
        ML, MT = 56, 60
        pages, cur, y = [], [], PH - MT
        for style, text in lines:
            size = {'h1': 15, 'h2': 11, 'mono': 8}.get(style, 9)
            lead = size + 4
            font = 'F2' if style == 'mono' else (
                'F3' if style in ('h1', 'h2') else 'F1')
            # Wrap on a conservative character width for the font size.
            width = int((PW - 2 * ML) / (size * 0.52))
            chunks = []
            for para in (text or ' ').split('\n'):
                while len(para) > width:
                    cut = para.rfind(' ', 0, width)
                    cut = cut if cut > 20 else width
                    chunks.append(para[:cut])
                    para = para[cut:].lstrip()
                chunks.append(para)
            for ch in chunks:
                if y < MT:
                    pages.append(cur)
                    cur, y = [], PH - MT
                cur.append(f'BT /{font} {size} Tf {ML} {y} Td '
                           f'({_pdf_escape(ch)}) Tj ET')
                y -= lead
            y -= 3
        pages.append(cur)

        objs, kids = [], []
        # 1=Catalog 2=Pages 3..=fonts, then per page content+page
        fonts = ('/F1 4 0 R /F2 5 0 R /F3 6 0 R')
        first_page_obj = 7
        for i, content in enumerate(pages):
            cid = first_page_obj + i * 2
            pid = cid + 1
            kids.append(f'{pid} 0 R')
            stream = '\n'.join(content).encode('latin-1', 'replace')
            objs.append((cid, b'<< /Length ' + str(len(stream)).encode()
                         + b' >>\nstream\n' + stream + b'\nendstream'))
            objs.append((pid, (
                f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PW} {PH}]'
                f' /Resources << /Font << {fonts} >> >>'
                f' /Contents {cid} 0 R >>').encode()))
        head = [
            (1, f'<< /Type /Catalog /Pages 2 0 R >>'.encode()),
            (2, (f'<< /Type /Pages /Kids [{" ".join(kids)}] '
                 f'/Count {len(pages)} >>').encode()),
            (3, b'<< /Title (' + _pdf_escape(title).encode('latin-1',
                                                           'replace')
             + b') >>'),
            (4, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'),
            (5, b'<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>'),
            (6, b'<< /Type /Font /Subtype /Type1 '
                b'/BaseFont /Helvetica-Bold >>'),
        ]
        allobjs = sorted(head + objs)
        out = bytearray(b'%PDF-1.4\n')
        offsets = {}
        for num, body in allobjs:
            offsets[num] = len(out)
            out += f'{num} 0 obj\n'.encode() + body + b'\nendobj\n'
        xref = len(out)
        n = max(offsets) + 1
        out += f'xref\n0 {n}\n'.encode()
        out += b'0000000000 65535 f \n'
        for i in range(1, n):
            out += f'{offsets.get(i, 0):010d} 00000 n \n'.encode()
        out += (f'trailer\n<< /Size {n} /Root 1 0 R /Info 3 0 R >>\n'
                f'startxref\n{xref}\n%%EOF\n').encode()
        with open(out_path, 'wb') as fh:
            fh.write(bytes(out))
        return True
    except Exception as exc:
        sys.stderr.write(f'[delivery] simple pdf failed: {exc}\n')
        return False


def _manifest_lines(fire_numbe: str, files: list, acq: dict,
                    stem: str) -> list:
    out = [('h1', f'ARCHIVE CONTENTS - {fire_numbe}'), ('p', '')]
    loc = (acq or {}).get('local')
    if loc is not None:
        src = 'L2 recent' if (acq or {}).get('source') == 'l2' else 'MRAP'
        out.append(('p', f'Imagery source: {src}'))
        if (acq or {}).get('time_unknown'):
            out.append(('p', 'Newest contributing acquisition: '
                             f'{loc:%Y-%m-%d} (time of day unavailable, '
                             'shown as 0000)'))
        else:
            out.append(('p', 'Newest contributing acquisition: '
                             f'{loc:%Y-%m-%d %H:%M %Z} local, '
                             f'{(acq or {}).get("utc")} UTC'))
        out.append(('p', f'Product naming: {stem}.*'))
        if (acq or {}).get('exact'):
            out.append(('p', 'This time is exact: the composite was '
                             'assembled by the application, which '
                             'recorded every contributing acquisition.'))
        else:
            out.append(('p', 'This time is an estimate. The MRAP '
                             'composite is built by a back-end process '
                             'that does not report which acquisitions '
                             'supplied which pixels, so this is the '
                             'newest acquisition available over the '
                             'covering tiles.'))
    out.append(('p', ''))
    out.append(('h2', 'Files'))
    for rel in files:
        d = describe(rel)
        if not d:
            continue
        out.append(('mono', rel))
        out.append(('p', '    ' + d))
    out.append(('p', ''))
    out.append(('p', 'Files ending in .hdr are ENVI headers describing '
                     'the raster beside them (dimensions, data type and '
                     'map projection) and are required to open it.'))
    return out


# ---------------------------------------------------------------------
# Prepared-download cache
# ---------------------------------------------------------------------
# The archive is built ahead of the click so the button can show its
# size and be disabled while stale. Keyed by a signature of everything
# that goes into it, so it rebuilds exactly when something changed and
# is reused otherwise.

def download_signature(result_dir: str, imagery=None) -> str:
    import hashlib
    h = hashlib.sha1()
    for root, _dirs, fnames in sorted(os.walk(result_dir)):
        for fn in sorted(fnames):
            if fn.startswith('.') or '.low.' in fn:
                continue
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            rel = os.path.relpath(p, result_dir)
            h.update(f'{rel}|{int(st.st_mtime)}|{st.st_size}|'.encode())
    for p in sorted(imagery or []):
        try:
            st = os.stat(p)
        except OSError:
            continue
        h.update(f'{os.path.basename(p)}|{int(st.st_mtime)}|'
                 f'{st.st_size}|'.encode())
    return h.hexdigest()[:16]


def cache_dir_for(output_root: str) -> str:
    d = os.path.join(output_root, '.download_cache')
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def cached_zip_path(output_root: str, fire_numbe: str, sig: str) -> str:
    return os.path.join(cache_dir_for(output_root),
                        f'{fire_numbe}__{sig}.zip')


def prune_cache(output_root: str, fire_numbe: str, keep: str = '') -> None:
    """Drop this fire's older archives; only the current one is useful."""
    import glob as _g
    for p in _g.glob(os.path.join(cache_dir_for(output_root),
                                  f'{fire_numbe}__*.zip')):
        if keep and os.path.abspath(p) == os.path.abspath(keep):
            continue
        try:
            os.remove(p)
        except OSError:
            pass
