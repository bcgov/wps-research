#!/usr/bin/env python3
"""
batch_fire_mapping/generate_report.py
=====================================
Generate a PDF report from fire mapping results.

Layout:
  Pass 1 — One full-page comparison image per fire (as large as possible).
  Pass 2 — Detail page per fire: image (left) + parameters table (right).

Standalone usage
----------------
    python batch_fire_mapping/generate_report.py  RESULTS_DIR  [--output report.pdf]

Called from run_fire_mapping.py
------------------------------
    generate_report(results_root)
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


def _find_png(fire_dir: str, suffix: str) -> str | None:
    """Return the path to <fire_numbe><suffix>.png if it exists."""
    fire_numbe = os.path.basename(fire_dir)
    path = os.path.join(fire_dir, f'{fire_numbe}{suffix}.png')
    return path if os.path.isfile(path) else None


def _escape_tex(s: str) -> str:
    """Escape characters that are special in LaTeX."""
    for ch, repl in [('\\', '/'), ('_', r'\_'), ('&', r'\&'),
                     ('%', r'\%'), ('#', r'\#'), ('{', r'\{'),
                     ('}', r'\}'), ('~', r'\textasciitilde{}')]:
        s = s.replace(ch, repl)
    return s


def _load_params(fire_dir: str) -> dict | None:
    """Load <fire_numbe>_params.yaml if it exists."""
    fire_numbe = os.path.basename(fire_dir)
    path = os.path.join(fire_dir, f'{fire_numbe}_params.yaml')
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _fire_size_ha(params: dict | None) -> float:
    """Extract fire_size_ha from params, returning 0.0 if unavailable."""
    if params is None:
        return 0.0
    return float(params.get('fire', {}).get('fire_size_ha', 0) or 0)


def _build_subtitle_tex(params: dict | None, fire_numbe: str) -> str:
    r"""Build LaTeX subtitle lines (joined by \\\\).

    Line 1: fire number | date range
    Line 2: Traditional estimation: X ha (Y m²)
    Line 3: Machine Learning estimation: X ha (Y m²)
    """
    # Line 1: fire number | dates
    parts = [_escape_tex(fire_numbe)]
    if params:
        acc = params.get('accumulation', {})
        start = acc.get('start_date')
        end = acc.get('end_date')
        if start and end:
            parts.append(f'{_escape_tex(str(start))} -- {_escape_tex(str(end))}')
    line1 = r' \quad|\quad '.join(parts)

    lines = [line1]

    if params:
        fire = params.get('fire', {})

        # Line 2: traditional estimation
        size_ha = float(fire.get('fire_size_ha', 0) or 0)
        if size_ha > 0:
            size_m2 = size_ha * 10000
            lines.append(
                f'Traditional estimation: {size_ha:,.1f} ha '
                f'({size_m2:,.0f} m$^2$)')

        # Line 3: ML estimation
        ml_ha = fire.get('ml_area_ha')
        ml_m2 = fire.get('ml_area_m2')
        if ml_ha is not None and ml_m2 is not None:
            lines.append(
                f'Machine Learning estimation: {ml_ha:,.1f} ha '
                f'({ml_m2:,.0f} m$^2$)')

    return r'\\[0.05cm]'.join(lines)


def _params_to_table_rows(params: dict) -> list[tuple[str, str, str]]:
    """Flatten params dict into (section, key, value) rows for the table."""
    rows = []
    display = [
        ('fire',       ['fire_numbe', 'fire_date', 'fire_size_ha']),
        ('crop',       ['width_px', 'height_px', 'total_px', 'padding']),
        ('inputs',     ['perimeter_type']),
        ('sampling',   ['actual_sample_size', 'seed']),
        ('accumulation', ['start_date', 'end_date']),
        ('tsne',       ['embed_bands', 'perplexity', 'learning_rate',
                        'max_iter', 'n_components', 'init']),
        ('hdbscan',    ['min_samples', 'controlled_ratio']),
        ('random_forest', ['n_estimators', 'max_depth', 'max_features']),
        ('class_brush',   ['brush_size', 'point_threshold']),
    ]
    for section, keys in display:
        section_data = params.get(section, {})
        if not section_data:
            continue
        for key in keys:
            if key in section_data and section_data[key] is not None:
                val = section_data[key]
                rows.append((section, key, str(val)))
    return rows


def _build_param_table_tex(params: dict) -> str:
    """Build a LaTeX tabular block from the params dict."""
    rows = _params_to_table_rows(params)
    if not rows:
        return ''

    lines = []
    lines.append(r'\scriptsize')
    lines.append(r'\renewcommand{\arraystretch}{1.2}')
    lines.append(r'\begin{tabular}[t]{@{}l l l@{}}')
    lines.append(r'\hline')
    lines.append(r'\textbf{Section} & \textbf{Parameter} & \textbf{Value} \\')
    lines.append(r'\hline')

    prev_section = None
    for section, key, val in rows:
        sec_label = _escape_tex(section) if section != prev_section else ''
        lines.append(
            f'{sec_label} & {_escape_tex(key)} & {_escape_tex(val)} \\\\'
        )
        prev_section = section

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


def generate_report(results_root: str, output_path: str = None) -> str | None:
    """
    Scan *results_root* for fire folders, collect PNGs and params,
    and build a PDF report.

    Returns path to the generated PDF, or None if generation failed.
    """
    fm_root = results_root

    if output_path is None:
        output_path = os.path.join(fm_root, 'report.pdf')

    # Discover fire directories (contain at least one PNG)
    fire_dirs = sorted([
        os.path.join(fm_root, d)
        for d in os.listdir(fm_root)
        if os.path.isdir(os.path.join(fm_root, d))
        and glob.glob(os.path.join(fm_root, d, '*.png'))
    ])

    if not fire_dirs:
        print('[report] No fire directories with PNGs found.')
        return None

    print(f'[report] Found {len(fire_dirs)} fire(s) with PNGs.')

    # Collect data for each fire
    fires = []
    for fire_dir in fire_dirs:
        fire_numbe = os.path.basename(fire_dir)
        comparison_png = _find_png(fire_dir, '_comparison')
        brush_png = _find_png(fire_dir, '_brush_comparison')
        no_viirs_png = _find_png(fire_dir, '_no_viirs')
        hero_png = comparison_png or brush_png or no_viirs_png
        if hero_png is None:
            continue
        fires.append({
            'fire_numbe': fire_numbe,
            'hero_png': hero_png,
            'brush_png': brush_png if brush_png and brush_png != hero_png else None,
            'params': _load_params(fire_dir),
        })

    if not fires:
        print('[report] No fires with usable PNGs.')
        return None

    # Sort by fire size descending (largest first)
    fires.sort(key=lambda f: _fire_size_ha(f['params']), reverse=True)

    # Build LaTeX document
    lines = [
        r'\documentclass[landscape]{article}',
        r'\usepackage{graphicx}',
        r'\usepackage[landscape,margin=0.5cm]{geometry}',
        r'\usepackage{array}',
        r'\pagestyle{empty}',
        r'\begin{document}',
    ]

    # ---- Pass 1: Full-page comparison images ----
    for i, fire in enumerate(fires):
        if i > 0:
            lines.append(r'\newpage')
        fire_label = _escape_tex(fire['fire_numbe'])
        subtitle = _build_subtitle_tex(fire['params'], fire['fire_numbe'])
        img_path = os.path.abspath(fire['hero_png']).replace('\\', '/')

        lines.append(r'\begin{center}')
        lines.append(r'{\Large\bfseries ' + fire_label + r'}\\[0.1cm]')
        lines.append(r'{\small ' + subtitle + r'}\\[0.2cm]')
        lines.append(
            r'\includegraphics[width=0.98\textwidth,'
            r'height=0.85\textheight,keepaspectratio]{'
            + img_path + r'}'
        )
        lines.append(r'\end{center}')

    # ---- Pass 2: Detail pages (image left + params right) ----
    for fire in fires:
        lines.append(r'\newpage')
        fire_label = _escape_tex(fire['fire_numbe'])
        subtitle = _build_subtitle_tex(fire['params'], fire['fire_numbe'])

        # Title
        lines.append(r'\noindent')
        lines.append(r'{\Large\bfseries Fire: ' + fire_label + r'}\\[0.1cm]')
        lines.append(r'{\small ' + subtitle + r'}\\[0.3cm]')

        # Two-column layout
        lines.append(r'\noindent')
        lines.append(r'\begin{minipage}[t]{0.60\textwidth}')

        img_path = os.path.abspath(fire['hero_png']).replace('\\', '/')
        lines.append(
            r'\includegraphics[width=\textwidth,'
            r'height=0.44\textheight,keepaspectratio]{'
            + img_path + r'}'
        )

        if fire['brush_png']:
            bp = os.path.abspath(fire['brush_png']).replace('\\', '/')
            lines.append(r'\\[0.2cm]')
            lines.append(
                r'\includegraphics[width=\textwidth,'
                r'height=0.40\textheight,keepaspectratio]{'
                + bp + r'}'
            )

        lines.append(r'\end{minipage}%')
        lines.append(r'\hfill')
        lines.append(r'\begin{minipage}[t]{0.37\textwidth}')

        if fire['params']:
            lines.append(_build_param_table_tex(fire['params']))
        else:
            lines.append(r'\textit{No parameters available.}')

        lines.append(r'\end{minipage}')

    lines.append(r'\end{document}')

    tex_content = '\n'.join(lines)

    # Write .tex and compile
    work_dir = tempfile.mkdtemp(prefix='fire_report_')
    tex_path = os.path.join(work_dir, 'report.tex')

    with open(tex_path, 'w') as f:
        f.write(tex_content)

    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode',
             '-shell-escape', tex_path],
            cwd=work_dir,
            capture_output=True,
            timeout=120,
        )
        pdf_tmp = os.path.join(work_dir, 'report.pdf')

        if not os.path.isfile(pdf_tmp):
            print('[report] pdflatex did not produce a PDF.')
            print(result.stdout.decode(errors='replace')[-1000:])
            return None

        shutil.copy2(pdf_tmp, output_path)
        print(f'[report] PDF report → {output_path}')
        return output_path

    except FileNotFoundError:
        print('[report] pdflatex not found — cannot generate PDF. '
              'Install texlive or equivalent.')
        return None
    except subprocess.TimeoutExpired:
        print('[report] pdflatex timed out.')
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# =========================================================================
# CLI entry point
# =========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        prog='generate_report.py',
        description='Generate a PDF report from fire mapping results.',
    )
    p.add_argument('results_dir',
                   help='Directory containing fire output folders')
    p.add_argument('--output', '-o', default=None,
                   help='Output PDF path (default: <results_dir>/report.pdf)')

    args = p.parse_args(argv)
    pdf = generate_report(args.results_dir, args.output)
    if pdf is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
