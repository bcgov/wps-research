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

KEEP_KEYS = {
    "envi",
}


def backup_once(path: str):
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(path, "rb") as f1, open(bak, "wb") as f2:
            f2.write(f1.read())


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ------------------------------------------------------------
# STRICT ENVI PARSER
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

        # brace block (map info / band names / etc.)
        if "{" in val and "}" not in val:
            buf = [val]
            i += 1
            while i < len(lines):
                buf.append(lines[i].rstrip("\n"))
                if "}" in lines[i]:
                    break
                i += 1
            val = "\n".join(buf)

        records.append((key, val))
        i += 1

    return records


def get_external_geo(date_str):
    path = Path(f"/data/mrap_bc/{date_str}_mrap.hdr")
    if not path.exists():
        raise FileNotFoundError(path)

    lines = read_lines(path)
    records = parse_envi(lines)

    geo = {}
    for k, v in records:
        if k in REMOVE_KEYS:
            geo[k] = v

    return geo


def extract_band_block(records):
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

    raw = m.group(1).strip()

    # split safely on commas ONLY
    parts = [p.strip().rstrip(",") for p in raw.split(",")]
    return [p for p in parts if p]


def rebuild_bands(bands):
    # IMPORTANT: NO newlines inside values (ENVI-safe compact form)
    inner = ",\n".join(bands)
    return f"band names = {{{inner}}}"


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
        if i < g:
            prefix = "pre "
        elif i < 2 * g:
            prefix = "pst "
        else:
            prefix = ""

        if not b.startswith("pre ") and not b.startswith("pst "):
            b = prefix + b

        out.append(b)

    return out


def clean_records(records):
    cleaned = []
    for k, v in records:
        if k in REMOVE_KEYS:
            continue
        if k == "band names":
            continue
        cleaned.append((k, v))
    return cleaned


def format_records(records):
    out = ["ENVI"]

    for k, v in records:
        if k == "envi":
            continue
        out.append(f"{k} = {v}")

    # remove empty lines & trailing whitespace
    out = [l.rstrip() for l in out if l.strip()]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    hdr_path = sys.argv[1]

    backup_once(hdr_path)

    lines = read_lines(hdr_path)
    records = parse_envi(lines)

    # -------------------------
    # BAND FIX
    # -------------------------
    band_block = extract_band_block(records)
    bands = parse_bands(band_block)

    if not bands:
        raise RuntimeError("No valid band names found")

    bands = prefix_bands(bands)
    new_band_block = rebuild_bands(bands)

    # -------------------------
    # CLEAN OLD RECORDS
    # -------------------------
    records = clean_records(records)

    # reinsert corrected band names ONCE
    records.append(("band names", new_band_block))

    # -------------------------
    # GEO OVERRIDE (NO APPEND BUG)
    # -------------------------
    date_str = re.search(r"(\d{8})", hdr_path).group(1)
    geo = get_external_geo(date_str)

    for k in REMOVE_KEYS:
        if k in geo:
            records.append((k, geo[k]))

    # -------------------------
    # WRITE OUTPUT
    # -------------------------
    out = format_records(records)
    write_text(hdr_path, out)


if __name__ == "__main__":
    main()
