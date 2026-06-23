'''20260622: stack file's header gets messed up sometimes.

terminate the band names field properly, and restore map / proj / CRS info from the corresponding post MRAP file.

Repair ENVI .hdr files for fire-mapping stacks.

Robustly parses even malformed headers (unclosed braces, spurious
newlines inside fields, duplicate geo keys), then:

1.  Strips ALL existing map info / projection info / coordinate system
    string records (however many, however malformed).
2.  Fixes the band names block: removes trailing commas before '}',
    strips stale pre/pst prefixes, applies the correct prefix rule
    (4-group if band count divisible by 4, else 3-group if divisible
    by 3).
3.  Re-inserts geo fields from the authoritative post-MRAP header
    (/data/mrap_bc/<yyyymmdd>_mrap.hdr), exactly once, in canonical
    order.
4.  Writes a clean header: ENVI marker, one newline between each
    record, NO trailing newline after the last record.

Usage:
    fire_mapping_stack_header_repair.py <target.hdr>
'''

import os
import re
import sys
from pathlib import Path

GEO_KEYS = frozenset({
    "map info",
    "projection info",
    "coordinate system string",
})

# Plausible ENVI key: one or more words of [A-Za-z0-9_], separated by
# single spaces.  This excludes quoted strings, PROJCS[...], numbers
# standing alone, etc., so it reliably detects when a new record starts
# inside a runaway (unclosed-brace) block.
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ ]*$")


# ── backup ────────────────────────────────────────────────────────────
def backup_once(path: str) -> None:
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(path, "rb") as src, open(bak, "wb") as dst:
            dst.write(src.read())


# ── parser ────────────────────────────────────────────────────────────
def _looks_like_new_record(line: str) -> bool:
    """True if *line* appears to start a fresh  key = value  record."""
    if "=" not in line:
        return False
    candidate = line.split("=", 1)[0].strip()
    return bool(_KEY_RE.match(candidate))


def parse_envi(text: str) -> list[tuple[str, str]]:
    """Return [(key, raw_value), ...] from an ENVI header.

    Handles:
      * multi-line brace-delimited values  (band names, CRS, …)
      * unclosed braces  (terminates the value when the next record
        starts, so a missing '}' doesn't swallow the rest of the file)
      * blank lines, the ENVI marker line, continuation-only lines
    """
    lines = text.split("\n")
    records: list[tuple[str, str]] = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        # skip blanks and the ENVI marker
        if not stripped or stripped == "ENVI":
            i += 1
            continue

        # every record must contain '='
        if "=" not in lines[i]:
            i += 1
            continue

        key_part, val_part = lines[i].split("=", 1)
        key = key_part.strip().lower()
        val = val_part.strip()
        i += 1

        # ── accumulate brace block ──
        if "{" in val:
            depth = val.count("{") - val.count("}")
            parts = [val]

            while i < len(lines) and depth > 0:
                # Before consuming the next line, check whether it
                # starts a brand-new record.  If so, the current brace
                # block was never closed — stop here.
                if _looks_like_new_record(lines[i]):
                    break

                nxt = lines[i]
                depth += nxt.count("{") - nxt.count("}")
                parts.append(nxt)
                i += 1

            val = "\n".join(parts)

        records.append((key, val))

    return records


# ── geo from reference ────────────────────────────────────────────────
def load_geo(date_str: str) -> dict[str, str]:
    ref = Path(f"/data/mrap_bc/{date_str}_mrap.hdr")
    if not ref.exists():
        raise FileNotFoundError(f"Reference header not found: {ref}")

    geo: dict[str, str] = {}
    for k, v in parse_envi(ref.read_text()):
        if k in GEO_KEYS:
            geo[k] = v
    return geo


# ── band names ────────────────────────────────────────────────────────
def extract_bands(raw: str) -> list[str]:
    """Parse individual band name strings out of a raw value.

    Tolerates:  missing closing '}', trailing comma, trailing period,
    extra whitespace / newlines between entries.
    """
    if not raw:
        return []

    # Strip everything up to and including the first '{'
    idx = raw.find("{")
    inner = raw[idx + 1:] if idx >= 0 else raw

    # Strip a trailing '}' if present
    idx2 = inner.rfind("}")
    if idx2 >= 0:
        inner = inner[:idx2]

    bands: list[str] = []
    for part in inner.split(","):
        cleaned = part.strip().rstrip(".").strip()
        if cleaned:
            bands.append(cleaned)

    return bands


def strip_prefix(name: str) -> str:
    """Remove any leading 'pre ' or 'pst ' (handles doubles too)."""
    while name.startswith("pre ") or name.startswith("pst "):
        name = name[4:]
    return name


def apply_prefixes(bands: list[str]) -> list[str]:
    """Apply pre / pst prefixes.

    Rule:
      divisible by 4  → group = 4
      divisible by 3  → group = 3  (only when not by 4)
      otherwise       → no prefixes

    First  *group*  bands get "pre ", next *group* get "pst ",
    the rest are left bare.
    """
    clean = [strip_prefix(b) for b in bands]
    n = len(clean)

    if n % 4 == 0:
        group = 4
    elif n % 3 == 0:
        group = 3
    else:
        return clean

    out: list[str] = []
    for i, b in enumerate(clean):
        if i < group:
            out.append(f"pre {b}")
        elif i < 2 * group:
            out.append(f"pst {b}")
        else:
            out.append(b)
    return out


def format_band_block(bands: list[str]) -> str:
    """band names = {entry1,\\nentry2,\\n…lastentry}"""
    inner = ",\n".join(bands)
    return f"band names = {{{inner}}}"


# ── output assembly ───────────────────────────────────────────────────
CORE_ORDER = [
    "samples",
    "lines",
    "bands",
    "header offset",
    "file type",
    "data type",
    "interleave",
    "byte order",
]

GEO_ORDER = [
    "map info",
    "projection info",
    "coordinate system string",
]


def build_header(records: list[tuple[str, str]],
                 band_block: str,
                 geo: dict[str, str]) -> str:
    """Assemble the final clean header text.

    Order:  ENVI  →  core fields  →  band names  →  other metadata
            →  geo (from reference).
    No trailing newline after the last line.
    """
    # Deduplicate non-band, non-geo keys  (last-write-wins)
    kv: dict[str, str] = {}
    for k, v in records:
        if k == "band names" or k in GEO_KEYS:
            continue
        kv[k] = v

    parts: list[str] = ["ENVI"]

    # core fields in stable order
    for k in CORE_ORDER:
        if k in kv:
            parts.append(f"{k} = {kv[k]}")

    # band names
    parts.append(band_block)

    # remaining metadata (preserve insertion order, skip core)
    core_set = set(CORE_ORDER)
    for k, v in kv.items():
        if k not in core_set:
            parts.append(f"{k} = {v}")

    # geo from reference, exactly once, canonical order
    for k in GEO_ORDER:
        if k in geo:
            parts.append(f"{k} = {geo[k]}")

    return "\n".join(parts)          # no trailing newline


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: fire_mapping_stack_header_repair.py <target.hdr>",
              file=sys.stderr)
        sys.exit(1)

    hdr = sys.argv[1]
    backup_once(hdr)

    text = Path(hdr).read_text()
    records = parse_envi(text)

    # ── bands ──
    raw_bands = None
    for k, v in records:
        if k == "band names":
            raw_bands = v
            break                     # first occurrence wins

    bands = extract_bands(raw_bands)
    if not bands:
        raise RuntimeError(f"No band names parsed from {hdr}")

    bands = apply_prefixes(bands)
    band_block = format_band_block(bands)

    # ── geo from reference ──
    m = re.search(r"(\d{8})", hdr)
    if not m:
        raise RuntimeError(f"Cannot extract yyyymmdd date from path: {hdr}")
    date_str = m.group(1)
    geo = load_geo(date_str)

    # ── write ──
    output = build_header(records, band_block, geo)
    Path(hdr).write_text(output)       # write_text adds no extra newline


if __name__ == "__main__":
    main()



