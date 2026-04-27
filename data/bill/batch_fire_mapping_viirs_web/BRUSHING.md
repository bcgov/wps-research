# Brushing — How `class_brush.exe` Cleans Up the Burn Mask

A full walk-through of the morphological post-processing stage that
turns the raw HDBSCAN classification into the final burn perimeter.
Source of truth: `cpp/class_brush.cpp` (~640 lines), wrapped by
`brush.py` in this package.

---

## 1. What problem brushing solves

HDBSCAN labels every pixel as either "burned" or "not burned" in the
2D t-SNE embedding. That label is **per-pixel and spatially blind** —
the clustering doesn't know that pixel (123,456) is a neighbour of
pixel (123,457). So even a perfect cluster boundary in embedding
space looks like *salt-and-pepper noise* on the actual map: stray
single-pixel "burned" labels in unburned regions, single-pixel
"unburned" holes inside the burn scar, and the burn scar itself often
broken into a main blob plus a fringe of small disconnected fragments.

Three problems to clean up:

1. **Tiny stray fragments** — a few isolated pixels labelled "burned"
   in an otherwise unburned area. Almost always misclassification noise.
2. **Burn-scar fragmentation** — the real burn comes out as one big
   blob plus several smaller pieces nearby. They should be one
   polygon, not many.
3. **Picking the answer** — once cleaned, you want a single binary
   raster: "this is the burn", not a labelled multi-class image.

`class_brush.exe` does all three in six pipeline stages.

---

## 2. Inputs and outputs

```
class_brush.exe <input_mask.bin> <brush_size> <point_threshold> [--all_segments]
```

| Argument | Meaning |
|---|---|
| `input_mask.bin` | Binary classification raster from HDBSCAN: `0` = unburned, non-zero = burned, `NaN` = no-data. ENVI float32. |
| `brush_size` | **Linking-window width in pixels.** Bigger window → more aggressive merging. Default 15. |
| `point_threshold` | **Minimum pixel count** for a component to survive. Default 10. |
| `--all_segments` | Optional. Without this flag (default), only the largest surviving component is kept. With it, every surviving component is written. |

Outputs (all ENVI float32):

| File | Stage that wrote it |
|---|---|
| `<input>_flood4.bin` | Stage 1 (flood-fill labels) |
| `<input>_flood4.bin_link.bin` | Stage 2 (after linking) |
| `<input>_flood4.bin_link.bin_recode.bin` | Stage 3 (contiguous labels) |
| `<input>_flood4.bin_link.bin_recode.bin_wheel.bin` | Stage 4 (RGB visualization) |
| One `.bin` per accepted component | Stage 6 (one-hot mask per component) |

The intermediate files are scratch artefacts. The final answer is
the per-component one-hot raster from Stage 6 — `brush.py` picks the
largest one (or OR's them together with `--all_segments`) and
promotes it back to `<FIRE>_crop.bin_classified.bin`.

---

## 3. The six stages, with code references

```
binary mask                                              0/1 + NaN
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 1 — flood-fill                    │   stage_flood
│  8-connected components                 │   class_brush.cpp:165
│  output: integer labels 1..K            │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 2 — link                          │   stage_link
│  union-find over a brush_size window    │   class_brush.cpp:220
│  merges components within proximity     │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 3 — recode                        │   stage_recode
│  renumber labels to 1..M contiguous     │   class_brush.cpp:276
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 4 — wheel (visualisation only)    │   stage_wheel
│  RGB image with shuffled hues per       │   class_brush.cpp:329
│  label, for QA. Not used by pipeline.   │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 5/6 — count + one-hot output      │   stage_onehot_output
│  drop labels < point_threshold,         │   class_brush.cpp:460
│  pick largest, write per-component      │
│  binary masks                           │
└─────────────────────────────────────────┘
   │
   ▼
binary mask                                              0/1 + NaN
(this is the final burn perimeter)
```

### Stage 1 — Flood-fill (`stage_flood`, lines 115–198)

Standard iterative flood-fill with **8-connectivity** (a pixel's
neighbours are all 8 surrounding pixels, including diagonals). The
fill distinguishes three pixel kinds:

- `NaN` → preserved as `NaN` (no-data passes through).
- `0.0` → background, label stays 0.
- non-zero → start of a new component, gets label `next_label`,
  flood-fills out to all 8-connected neighbours that share the same
  input value.

Output: a float32 raster where each connected blob has a unique
positive integer label. After this stage, salt-and-pepper noise
shows up as many tiny components.

**Important detail**: the fill uses an explicit stack
(`vector<size_t>`) rather than recursion to avoid stack overflow on
large connected regions. Visited tracking is a separate `uint8_t`
array.

### Stage 2 — Link (`stage_link`, lines 220–269)

This is the heart of brushing and the one most people misunderstand.
The goal: merge components that are *near each other* even if they
don't physically touch.

Mechanism:

1. **Union-find** initialised with one set per component (label).
2. **Sliding window** of size `nwin × nwin` (where `nwin = brush_size`),
   stepping by `frac = nwin / 2` — so windows overlap by half their
   width.
3. For each window position, collect every distinct non-zero,
   non-NaN label in that window. If more than one label appears in
   the same window, **union all those labels into one set**.
4. After all windows are processed, replace each pixel's label with
   the root of its union-find set.

Effect: if two components are within roughly `brush_size` pixels of
each other in any direction, they end up sharing a label. The
"distance" the linking window sees is bounded — diagonally separated
components may need to fall in the same window to get merged.

**Why this is monotone in `brush_size`**: union-find never splits
sets. A bigger window is a superset of a smaller window's
observations, so any merge that happens at `brush_size = 10` also
happens at `brush_size = 20`. Components only get *more* merged as
`brush_size` grows.

**Code subtlety**: labels are stored as `float` keys in
`unordered_map<float, float>`. Integer-valued floats are exact, so
key lookup is reliable, but the file deliberately re-validates with
`guard_integer_label()` in case a stray fractional value appears.

### Stage 3 — Recode (`stage_recode`, lines 276–299)

After linking, surviving labels are an arbitrary subset of the
original 1..K (e.g. `{3, 7, 12, 15}` if linking merged everything
else). This stage just renumbers them to `1..M` contiguous, in order
of their original numeric value. Background `0` and `NaN` pass
through unchanged.

Purely cosmetic — needed because Stage 6 iterates `for N = 1 to
n_classes`.

### Stage 4 — Wheel (`stage_wheel`, lines 329 onward)

Writes a 3-band RGB float32 raster where each component gets a
distinct hue from a shuffled wheel. **Not used downstream** — it's a
QA aid for inspecting the labelling visually in any ENVI viewer.

### Stage 5/6 — Count + one-hot (`stage_onehot_output`, lines 460–540)

This is where `point_threshold` and `--all_segments` come in.

1. **Pre-scan** (lines 469–475): count pixels per label. No mask
   allocations yet; just a pass over the recoded raster.
2. **Pick the main segment** (lines 478–485): the largest component
   whose pixel count meets `point_threshold`. If no component meets
   the threshold, print "No components found above threshold" and
   exit — the input is empty.
3. **Loop over labels 1..M** (line 494):
   - If the count is below `point_threshold`, **skip** (component
     is noise).
   - If `--all_segments` is off (default) and this isn't the main
     segment, **skip**.
   - Otherwise, build a one-hot binary mask (1 where pixel == this
     label, 0 elsewhere, NaN preserved) and write it to disk.
4. Each accepted component gets its own `.bin` + `.hdr` pair, and a
   line is printed to stdout: `+component <N> <pixel_count>`.

So the default behaviour is **"keep only the largest component above
threshold"** — exactly what you want for a single fire scar.
`--all_segments` is for cases where multiple disjoint scars are
expected.

---

## 4. The two parameters, in plain English

### `brush_size` (default 15)

**What it controls**: how aggressively nearby fragments get merged
into one component during Stage 2.

**Effect of raising it**: more merging. Components that were separate
become one. The largest component grows or stays the same.

**Effect of lowering it**: less merging. The largest component shrinks
or stays the same.

**Mental model**: imagine each surviving fragment as a magnet, and
`brush_size` as the magnet's reach. Bigger reach → more fragments
clump together.

### `point_threshold` (default 10)

**What it controls**: minimum pixel count for a component to survive
Stage 5/6.

**Effect of raising it**: stricter. More small components get dropped
as noise. The result polygon shrinks or stays the same.

**Effect of lowering it**: more permissive. Small components survive.
The result polygon grows or stays the same (when `--all_segments`
is on; otherwise only the *largest* surviving component matters, so
lowering this only changes what's eligible to *be* the largest).

### Interaction

These two are not independent:

- **High `brush_size` + low `point_threshold`**: many small fragments
  get linked into the main blob, and any leftover small ones survive.
  Maximally inclusive — the burn polygon expands to include even
  marginal pixels.
- **Low `brush_size` + high `point_threshold`**: only the genuinely
  large, contiguous burn blob survives. Conservative — the polygon
  is tight around the densest part of the scar.
- **Default (15, 10)**: a middle ground that handles typical
  Sentinel-2 burn scars on 30 m pixels — `brush_size = 15` reaches
  ~450 m, enough to bridge most fragmentation; `point_threshold = 10`
  drops anything smaller than ~9 000 m² (about 1 hectare).

### Monotonicity (your earlier question)

| Action | Effect on burn polygon area |
|---|---|
| Raise `brush_size` | grows or stays the same |
| Lower `brush_size` | shrinks or stays the same |
| Raise `point_threshold` | shrinks or stays the same |
| Lower `point_threshold` | grows or stays the same |

Each parameter is monotone *in isolation*. The exception, when both
move at once: a low `brush_size` can split a single component into
two, both of which individually clear `point_threshold` — so total
burned-pixel count stays the same, but the polygon is now two
pieces. With `--all_segments` off, only one piece is kept, so the
polygon area can drop sharply. This is the only "non-monotone"
edge case worth knowing about.

---

## 5. How the package consumes brushing

`brush.py::_run_class_brush_only` is the wrapper. It:

1. Spawns `class_brush.exe` as a subprocess via `_stream_subprocess`,
   so its stdout streams into the per-fire console log and the
   `_rebrush_procs` registry can SIGTERM it on cancel.
2. Parses the `+component N <pixel_count>` lines to know which
   one-hot rasters were produced.
3. Either picks the **largest** component file or **OR's all** of
   them (`brush_all_segments`) into a single binary raster.
4. Returns the binary raster to the caller, plus a "cancelled" flag.

The caller (`handlers/rebrush.py` for rebrushing, `workers.py` for
mapping) then:

- Saves the pre-brush mask as `*_classified_raw.bin` if it doesn't
  exist yet (so a subsequent rebrush can re-start from the
  HDBSCAN output, not from an already-brushed mask).
- Overwrites `*_classified.bin` with the brushed result.
- Regenerates `_brush_comparison.png` (raw vs. brushed side-by-side)
  and `_comparison.png` (perimeter overlay on RGB).
- Updates `fire.last_params` with `brush_size`, `point_threshold`,
  `brush_all_segments` so a later accept persists them.

For the **post-accept rebrush** flow (the recent change), the mask
is staged to a per-run filename instead of overwriting the canonical
one, and a gallery entry is appended — same `class_brush.exe`
mechanics, different file destinations.

---

## 6. Edge cases the C++ guards against

- **NaN propagation**: every stage preserves `NaN` pixels untouched.
  No-data never gets accidentally relabelled.
- **Non-integer labels**: `guard_integer_label` exits with an error
  if a label is not an integer-valued float. Defends against
  upstream bugs that might write fractional values.
- **Multi-label one-hot**: `stage_count_onehot` aborts if a
  supposedly one-hot mask contains more than one distinct non-zero
  label. Defends against bugs in Stage 6's mask construction.
- **Allocation overflow**: `mul_overflow` checks `nrow × ncol` for
  size_t overflow on extremely large rasters before allocation.
- **Zero components above threshold**: returns cleanly with a
  message; doesn't crash, doesn't write a bogus output.

---

## 7. What brushing does NOT do

It's worth being explicit about the boundary, because morphological
operations that *sound* similar are absent:

- **No dilation / erosion** — the burn polygon's *outline* is not
  expanded or contracted by `brush_size`. Only component *linking*
  happens. A lone burned pixel surrounded by unburned pixels stays
  exactly that one pixel (until `point_threshold` deletes it).
- **No hole-filling** — small unburned holes inside a burn scar
  remain unburned. If you want them filled, you'd need a separate
  morphological closing.
- **No edge smoothing** — pixel-stair-step edges from HDBSCAN stay
  pixel-stair-step. The polygonize stage that follows (when
  exporting to KML / shapefile) is what produces the polygon vertex
  list, but it doesn't smooth either; for that you'd run
  `ogr2ogr -simplify` on the polygonized output.
- **No reclassification** — brushing operates on *which pixels are
  burned*, not *what kind of burn*. Severity and intensity are
  outside the brush's scope.

If you need any of those, they belong upstream (in HDBSCAN tuning)
or downstream (in vector simplification), not in `class_brush.exe`.
