'''20260622: stack file's header gets messed up sometimes.

terminate the band names field properly, and restore map / proj / CRS info from the corresponding post MRAP file.
'''
#!/usr/bin/env python3
import sys
import os
import re
from pathlib import Path


REMOVE_KEYS = {
    "map info",
    "projection info",
    "coordinate system string",
}


# ------------------------------------------------------------
# BACKUP (ONCE ONLY)
# ------------------------------------------------------------
def backup_once(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(path, "rb") as f1, open(bak, "wb") as f2:
            f2.write(f1.read())


# ------------------------------------------------------------
# ENVI PARSER (STATE MACHINE, BRACE SAFE)
# ------------------------------------------------------------
def parse_envi(lines):
    records = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip("\n")

        if not line.strip():
            i += 1
            continue

        if "=" not in line:
            i += 1
            continue

        key, val = line.split("=", 1)
        key = key.strip().lower()
        val = val.strip()

        # ---- capture full {...} block safely ----
        if "{" in val:
            depth = val.count("{") - val.count("}")

            buf = [val]
            i += 1

            while i < len(lines) and depth > 0:
                l = lines[i].rstrip("\n")
                depth += l.count("{") - l.count("}")
                buf.append(l)
                i += 1

            val = "\n".join(buf)

        records.append((key, val))
        i += 1

    return records


# ------------------------------------------------------------
# GEO EXTRACTION (AUTHORITATIVE SOURCE)
# ------------------------------------------------------------
def load_geo(date_str):
    path = Path(f"/data/mrap_bc/{date_str}_mrap.hdr")
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text().splitlines()
    records = parse_envi(lines)

    geo = {}
    for k, v in records:
        if k in REMOVE_KEYS:
            geo[k] = v

    return geo


# ------------------------------------------------------------
# BAND HANDLING (FIXED - NO PREFIX BUGS)
# ------------------------------------------------------------
def extract_band(records):
    for k, v in records:
        if k == "band names":
            return v
    return None


def parse_band_list(block):
    if not block:
        return []

    m = re.search(r"\{(.*)\}", block, re.S)
    if not m:
        return []

    raw = m.group(1).strip()

    # split on commas ONLY
    parts = [p.strip().rstrip(",") for p in raw.split(",")]
    return [p for p in parts if p]


def prefix_bands(bands):
    n = len(bands)

    if n % 4 == 0:
        g = 4
    elif n % 3 == 0:
        g = 3
    else:
        return bands

    out = []
    for i, b in enumerate(bands):
        prefix = ""
        if i < g:
            prefix = "pre "
        elif i < 2 * g:
            prefix = "pst "

        # avoid double-prefixing
        if not b.startswith("pre ") and not b.startswith("pst "):
            b = prefix + b

        out.append(b)

    return out


def build_band_block(bands):
    # ENVI-safe compact formatting (no duplicate key injection)
    return "band names = {" + ",\n".join(bands) + "}"


# ------------------------------------------------------------
# CLEAN + DEDUP RECORDS (CRITICAL FIX)
# ------------------------------------------------------------
def normalize(records):
    """
    Enforces:
    - only ONE instance of each key
    - last occurrence wins
    """
    latest = {}

    for k, v in records:
        if k == "band names":
            continue
        if k in REMOVE_KEYS:
            continue
        latest[k] = v

    return latest


def format_envi(kv, band_block, geo):
    out = ["ENVI"]

    # core metadata first (stable ordering)
    core_order = [
        "samples",
        "lines",
        "bands",
        "header offset",
        "file type",
        "data type",
        "interleave",
        "byte order",
    ]

    for k in core_order:
        if k in kv:
            out.append(f"{k} = {kv[k]}")

    # band names (fixed)
    out.append(band_block)

    # everything else except geo
    for k, v in kv.items():
        if k in core_order:
            continue
        out.append(f"{k} = {v}")

    # authoritative geo (ONLY ONCE)
    for k in REMOVE_KEYS:
        if k in geo:
            out.append(f"{k} = {geo[k]}")

    return "\n".join(out).rstrip() + "\n"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    hdr = sys.argv[1]

    backup_once(hdr)

    lines = Path(hdr).read_text().splitlines()
    records = parse_envi(lines)

    # -----------------------
    # BAND FIX
    # -----------------------
    band_block = extract_band(records)
    bands = parse_band_list(band_block)

    if not bands:
        raise RuntimeError("Failed to parse band names")

    bands = prefix_bands(bands)
    band_block = build_band_block(bands)

    # -----------------------
    # DEDUP EVERYTHING
    # -----------------------
    kv = normalize(records)

    # -----------------------
    # GEO OVERRIDE (AUTHORITATIVE)
    # -----------------------
    date_str = re.search(r"(\d{8})", hdr).group(1)
    geo = load_geo(date_str)

    # -----------------------
    # WRITE OUTPUT
    # -----------------------
    out = format_envi(kv, band_block, geo)
    Path(hdr).write_text(out)


if __name__ == "__main__":
    main()

