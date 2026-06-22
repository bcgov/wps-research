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
# BACKUP
# ------------------------------------------------------------
def backup_once(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(path, "rb") as f1, open(bak, "wb") as f2:
            f2.write(f1.read())


# ------------------------------------------------------------
# PARSER (brace-safe)
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

        # brace block capture (map info / band names / CRS)
        if "{" in val and "}" not in val:
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
# GEO LOADING (authoritative)
# ------------------------------------------------------------
def load_geo(date_str):
    path = Path(f"/data/mrap_bc/{date_str}_mrap.hdr")
    if not path.exists():
        raise FileNotFoundError(path)

    return parse_envi(path.read_text().splitlines())


def geo_dict(records):
    geo = {}
    for k, v in records:
        if k in REMOVE_KEYS:
            geo[k] = v
    return geo


# ------------------------------------------------------------
# BAND HANDLING (FIXED RULES)
# ------------------------------------------------------------
def extract_band(records):
    for k, v in records:
        if k == "band names":
            return v
    return None


def parse_bands(block):
    if not block:
        return []

    m = re.search(r"\{(.*)\}", block, re.S)
    if not m:
        return []

    raw = m.group(1)

    # FIX: handle commas + accidental periods safely
    parts = []
    for p in raw.replace("\n", " ").split(","):
        p = p.strip().rstrip(".").rstrip(",")
        if p:
            parts.append(p)

    return parts


# ------------------------------------------------------------
# EXACT PREFIX RULE IMPLEMENTATION
# ------------------------------------------------------------
def prefix_bands(bands):
    n = len(bands)

    if n % 4 == 0:
        group = 4
    elif n % 3 == 0:
        group = 3
    else:
        return bands

    out = []
    for i, b in enumerate(bands):

        if i < group:
            prefix = "pre "
        elif i < 2 * group:
            prefix = "pst "
        else:
            prefix = ""

        # prevent double prefixing
        if b.startswith("pre ") or b.startswith("pst "):
            out.append(b)
        else:
            out.append(prefix + b)

    return out


def build_band_block(bands):
    # IMPORTANT: ENVI expects single-line per entry, NO duplicate key
    inner = ",\n".join(bands)
    return f"band names = {{{inner}}}"


# ------------------------------------------------------------
# CLEAN RECORDS (remove duplicates safely)
# ------------------------------------------------------------
def normalize(records):
    kv = {}

    for k, v in records:
        if k == "band names":
            continue
        if k in REMOVE_KEYS:
            continue
        kv[k] = v  # last-write-wins

    return kv


# ------------------------------------------------------------
# OUTPUT WRITER (ENSURES GEO PRESERVED EXACTLY ONCE)
# ------------------------------------------------------------
def format_output(kv, band_block, geo):
    out = ["ENVI"]

    # stable ordering for core fields
    core = [
        "samples",
        "lines",
        "bands",
        "header offset",
        "file type",
        "data type",
        "interleave",
        "byte order",
    ]

    for k in core:
        if k in kv:
            out.append(f"{k} = {kv[k]}")

    out.append(band_block)

    # rest of metadata (non-geo)
    for k, v in kv.items():
        if k not in core:
            out.append(f"{k} = {v}")

    # FIX: ensure geo is appended EXACTLY ONCE, correct order
    for k in ["map info", "projection info", "coordinate system string"]:
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

    # band fix
    band_block = extract_band(records)
    bands = parse_bands(band_block)

    if not bands:
        raise RuntimeError("No band names parsed")

    bands = prefix_bands(bands)
    band_block = build_band_block(bands)

    # normalize metadata
    kv = normalize(records)

    # geo override (authoritative source)
    date_str = re.search(r"(\d{8})", hdr).group(1)
    geo_records = load_geo(date_str)
    geo = geo_dict(geo_records)

    # write
    out = format_output(kv, band_block, geo)
    Path(hdr).write_text(out)


if __name__ == "__main__":
    main()
