/*
 * class_brush.cpp  —  Combined fire-boundary tracing pipeline
 *
 * Replaces: flood.exe, class_link.exe, class_recode.exe,
 *           class_wheel.exe, class_count.exe
 *
 * Usage:
 *   class_brush.exe <input_mask.bin> <brush_size> <point_threshold>
 *
 * Parameters:
 *   input_mask.bin   : single-band float32 (ENVI type 4) binary mask.
 *                      Foreground pixels have value 1.0, background 0.0,
 *                      no-data areas are NaN.
 *   brush_size       : linking window width in pixels (e.g. 111).
 *                      Nearby connected components within this window
 *                      radius are merged together.
 *   point_threshold  : minimum pixel count for a component to be accepted
 *                      (e.g. 10). Components below this are silently dropped.
 *
 * Pipeline stages (all in-memory, no intermediate files):
 *   1. flood    — connected-component labelling (8-connectivity, NaN=nodata)
 *   2. link     — merge nearby labels using a sliding window (union-find)
 *   3. recode   — make labels contiguous starting from 1
 *                 (0 is reserved for background / no-data)
 *   4. onehot   — split recoded map into per-component binary masks
 *   5. count    — count pixels per component; skip if below threshold
 *   6. wheel    — write colour-wheel visualisation of the recoded label map
 *
 * Outputs (written to same directory as input):
 *   <input>_flood4.bin / .hdr          — raw flood-fill labels (float32)
 *   <input>_flood4.bin_link.bin / .hdr — after window-link merge (float32)
 *   <input>_flood4.bin_link.bin_recode.bin / .hdr — contiguous labels
 *   <input>_flood4.bin_link.bin_recode.bin_wheel.bin / .hdr — RGB vis
 *   <input>_comp_NNN.bin / .hdr        — per-component binary masks
 *                                        (one file per accepted component,
 *                                         NNN = zero-padded component index)
 *
 * Stdout (one line per accepted component, parsed by class_brush.py):
 *   +component <N> <pixel_count>
 *
 * Notes:
 *   • class_link uses float as map key. Labels are integer-valued floats
 *     written/read from binary — equality comparisons are safe in this
 *     context. A guard is included to error on non-integer label values.
 *   • Label 0 is reserved for background throughout all stages.
 *   • NaN pixels propagate as NaN through all stages.
 *   • class_wheel shuffle: shuf has exactly N entries for N distinct
 *     non-zero labels, indexed 0..N-1. Label d (1..N) maps to
 *     shuf[(int)d-1], which is always in range.
 *
 * Dependencies: misc.h (same as original individual tools)
 *
 * Build:
 *   g++ -O2 -std=c++17 -o class_brush.exe class_brush.cpp
 *
 * 20230712 original individual tools
 * 20251022 generalised flood fill
 * combined + fixed: 2026
 */

#include "misc.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <stdexcept>

using std::size_t;
using std::string;
using std::map;
using std::set;
using std::vector;
using std::unordered_map;
using std::unordered_set;

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

static inline bool mul_overflow(size_t a, size_t b) {
    if (a == 0 || b == 0) return false;
    return a > (std::numeric_limits<size_t>::max() / b);
}

/** Guard: label values must be non-negative integers (or NaN) for safe
 *  use as float map keys. Throws std::runtime_error otherwise. */
static void guard_integer_label(float v, const char *stage) {
    if (std::isnan(v)) return;
    if (v < 0.f || v != std::floor(v) || v > 1e7f) {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "%s: non-integer or out-of-range label value %f — "
            "float map-key equality would be unreliable.", stage, (double)v);
        throw std::runtime_error(buf);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 1: Flood fill — connected-component labelling
// ═══════════════════════════════════════════════════════════════════════════
// Labels: 0 = background (input value 0), NaN = no-data, 1..K = components.
// 8-connectivity. Returns float array (caller frees).

static long int flood_iter(float *dat,
                            float *out_f,
                            uint8_t *visited,
                            size_t nrow, size_t ncol,
                            long int si, long int sj,
                            float label_val,
                            size_t next_label)
{
    if (si < 0 || sj < 0 || (size_t)si >= nrow || (size_t)sj >= ncol) return 0;
    size_t start_idx = (size_t)si * ncol + (size_t)sj;
    if (visited[start_idx]) return 0;
    float vstart = dat[start_idx];
    if (std::isnan(vstart) || vstart != label_val) return 0;

    long int count = 0;
    vector<size_t> stack;
    stack.reserve(1024);
    stack.push_back(start_idx);

    while (!stack.empty()) {
        size_t idx = stack.back(); stack.pop_back();
        if (visited[idx]) continue;
        float v = dat[idx];
        if (std::isnan(v) || v != label_val) continue;
        visited[idx] = 1;
        out_f[idx] = static_cast<float>(next_label);
        count++;
        size_t ii = idx / ncol;
        size_t jj = idx % ncol;
        for (int di = -1; di <= 1; ++di) {
            long ni = (long)ii + di;
            if (ni < 0 || (size_t)ni >= nrow) continue;
            for (int dj = -1; dj <= 1; ++dj) {
                if (di == 0 && dj == 0) continue;
                long nj = (long)jj + dj;
                if (nj < 0 || (size_t)nj >= ncol) continue;
                size_t nidx = (size_t)ni * ncol + (size_t)nj;
                if (!visited[nidx]) {
                    float nv = dat[nidx];
                    if (!std::isnan(nv) && nv == label_val)
                        stack.push_back(nidx);
                }
            }
        }
    }
    return count;
}

/** Run flood fill on dat[np]. Returns allocated float array of labels.
 *  Label 0 = background (input 0.0), NaN = no-data, 1..K = components. */
static float *stage_flood(float *dat, size_t nrow, size_t ncol, size_t &n_labels) {
    size_t np = nrow * ncol;
    float  *out_f   = falloc(np);
    uint8_t *visited = (uint8_t *)alloc(np * sizeof(uint8_t));
    memset(visited, 0, np * sizeof(uint8_t));
    memset(out_f,   0, np * sizeof(float));

    size_t next_label = 1;

    for (long int i = 0; i < (long int)nrow; ++i) {
        for (long int j = 0; j < (long int)ncol; ++j) {
            size_t ij = (size_t)i * ncol + (size_t)j;
            if (visited[ij]) continue;
            float val = dat[ij];
            if (std::isnan(val)) {
                visited[ij] = 1;
                out_f[ij]   = NAN;
                continue;
            }
            if (val == 0.f) {
                // background — mark visited, leave label 0
                visited[ij] = 1;
                out_f[ij]   = 0.f;
                continue;
            }
            long int flooded = flood_iter(dat, out_f, visited,
                                          nrow, ncol, i, j, val, next_label);
            if (flooded > 0) next_label++;
        }
    }
    n_labels = next_label - 1;
    free(visited);
    return out_f;
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 2: Link — merge nearby labels with union-find (sliding window)
// ═══════════════════════════════════════════════════════════════════════════

static float uf_find(unordered_map<float, float> &p, float x) {
    if (p[x] == x) return x;
    p[x] = uf_find(p, p[x]);  // path compression
    return p[x];
}

static bool uf_unite(unordered_map<float, float> &p, float x, float y) {
    x = uf_find(p, x);
    y = uf_find(p, y);
    if (x == y) return false;
    if (x < y) p[y] = x;
    else        p[x] = y;
    return true;
}

/** Link nearby labels. Returns new allocated float array. */
static float *stage_link(float *dat, size_t nrow, size_t ncol, long int nwin) {
    size_t np = nrow * ncol;
    float *out = falloc(np);

    unordered_map<float, float> p;
    p.reserve(4096);

    // Initialise union-find: each non-zero, non-NaN label is its own root
    for (size_t i = 0; i < np; ++i) {
        float d = dat[i];
        if (!std::isnan(d) && d > 0.f) {
            guard_integer_label(d, "stage_link init");
            if (p.count(d) == 0) p[d] = d;
        }
    }

    long int frac = nwin / 2;
    unordered_set<float> merge;
    merge.reserve(256);

    for (long int i = 0; i < (long int)nrow + frac; i += frac) {
        for (long int j = 0; j < (long int)ncol + frac; j += frac) {
            merge.clear();
            for (long int di = 0; di < nwin; ++di) {
                long int ii = i + di;
                if (ii >= (long int)nrow) continue;
                for (long int dj = 0; dj < nwin; ++dj) {
                    long int jj = j + dj;
                    if (jj >= (long int)ncol) continue;
                    float d = dat[(size_t)ii * ncol + (size_t)jj];
                    if (!std::isnan(d) && d > 0.f) merge.insert(d);
                }
            }
            if (merge.size() > 1) {
                float parent = *(merge.begin());
                for (auto it = merge.begin(); it != merge.end(); ++it) {
                    if (it != merge.begin()) uf_unite(p, parent, *it);
                }
            }
        }
    }

    for (size_t i = 0; i < np; ++i) {
        float d = dat[i];
        if (std::isnan(d))  out[i] = NAN;
        else if (d == 0.f)  out[i] = 0.f;
        else                out[i] = uf_find(p, d);
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 3: Recode — make labels contiguous starting at 1
// ═══════════════════════════════════════════════════════════════════════════
// 0 remains 0 (background), NaN remains NaN, non-zero labels → 1..M.

static float *stage_recode(float *dat, size_t np, size_t &n_classes) {
    map<float, size_t> count;
    for (size_t i = 0; i < np; ++i) {
        float d = dat[i];
        if (std::isnan(d) || d == 0.f) continue;
        guard_integer_label(d, "stage_recode");
        count[d]++;
    }

    // Assign contiguous labels starting at 1 (0 reserved for background)
    float ci = 1.f;
    map<float, float> lut;
    for (auto &kv : count) lut[kv.first] = ci++;
    n_classes = (size_t)(ci - 1.f);

    float *out = falloc(np);
    for (size_t i = 0; i < np; ++i) {
        float d = dat[i];
        if (std::isnan(d))  out[i] = NAN;
        else if (d == 0.f)  out[i] = 0.f;
        else                out[i] = lut[d];
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 4: Colour-wheel visualisation
// ═══════════════════════════════════════════════════════════════════════════
// Writes a 3-band RGB float32 file. Label 0 → black, labels 1..N → hue wheel
// with shuffled assignment to avoid adjacent labels having similar colours.
//
// Shuffle fix: shuf has exactly N entries (indices 0..N-1).
// Label d (1..N) → shuf[(int)d - 1], always in range.

static void hsv_to_rgb_local(float *r, float *g, float *b,
                               float h, float s, float v)
{
    // h in [0,360), s,v in [0,1]
    int   hi = (int)(h / 60.f) % 6;
    float f  = h / 60.f - std::floor(h / 60.f);
    float p  = v * (1.f - s);
    float q  = v * (1.f - f * s);
    float t  = v * (1.f - (1.f - f) * s);
    switch (hi) {
        case 0: *r=v; *g=t; *b=p; break;
        case 1: *r=q; *g=v; *b=p; break;
        case 2: *r=p; *g=v; *b=t; break;
        case 3: *r=p; *g=q; *b=v; break;
        case 4: *r=t; *g=p; *b=v; break;
        default:*r=v; *g=p; *b=q; break;
    }
}

static void stage_wheel(float *dat, size_t nrow, size_t ncol,
                         size_t n_classes,
                         const str &ofn, const str &ohfn)
{
    size_t np = nrow * ncol;

    // Count distinct non-zero, non-NaN labels (should be 1..n_classes after recode)
    map<float, size_t> count;
    for (size_t i = 0; i < np; ++i) {
        float d = dat[i];
        if (!std::isnan(d) && d != 0.f) count[d]++;
    }
    size_t N = count.size();  // number of non-zero labels

    // Build colour table: label → (r,g,b) based on hue position
    map<float, float> code_r, code_g, code_b;
    {
        // Use a priority queue (descending) to visit labels in consistent order
        // (matches original behaviour)
        std::priority_queue<float> pq;
        for (auto &kv : count) pq.push(kv.first);

        long int ci = 0;
        float denom = (N > 1) ? (float)(N - 1) : 1.f;
        while (!pq.empty()) {
            float d = pq.top(); pq.pop();
            if (d == 0.f) {
                code_r[d] = code_g[d] = code_b[d] = 0.f;
            } else {
                float h = 360.f * (float)ci / denom;
                float r, g, b;
                hsv_to_rgb_local(&r, &g, &b, h, 1.f, 1.f);
                code_r[d] = r;
                code_g[d] = g;
                code_b[d] = b;
                ci++;
            }
        }
    }

    // Build shuffle: N entries, indices 0..N-1.
    // Label d (1..N after recode) uses shuf[(int)d - 1].
    // shuf contains the label values 1..N (as floats), shuffled.
    vector<float> shuf;
    shuf.reserve(N);
    for (auto &kv : count) {
        if (kv.first != 0.f) shuf.push_back(kv.first);
    }
    // shuf.size() == N; labels are 1-based so shuf[(int)d-1] is index 0..N-1.
    unsigned seed = 1;
    std::shuffle(shuf.begin(), shuf.end(), std::default_random_engine(seed));

    // Build final colour lookup: label → shuffled colour
    map<float, float> c_r, c_g, c_b;
    for (auto &kv : code_r) {
        float d = kv.first;
        if (d == 0.f) {
            c_r[d] = c_g[d] = c_b[d] = 0.f;
        } else {
            int idx = (int)d - 1;
            if (idx < 0 || (size_t)idx >= N) {
                fprintf(stderr, "stage_wheel: label %f out of shuf range (N=%zu)\n",
                        (double)d, N);
                // Fallback: black
                c_r[d] = c_g[d] = c_b[d] = 0.f;
                continue;
            }
            float sv = shuf[idx];
            c_r[d] = code_r[sv];
            c_g[d] = code_g[sv];
            c_b[d] = code_b[sv];
        }
    }

    // Write 3-band interleaved-by-band float32 RGB file
    hwrite(ohfn, nrow, ncol, 3);
    FILE *outf = fopen(ofn.c_str(), "wb");
    if (!outf) { fprintf(stderr, "stage_wheel: cannot open %s\n", ofn.c_str()); return; }
    for (size_t i = 0; i < np; ++i) {
        float v = std::isnan(dat[i]) ? 0.f : c_r[dat[i]];
        fwrite(&v, sizeof(float), 1, outf);
    }
    for (size_t i = 0; i < np; ++i) {
        float v = std::isnan(dat[i]) ? 0.f : c_g[dat[i]];
        fwrite(&v, sizeof(float), 1, outf);
    }
    for (size_t i = 0; i < np; ++i) {
        float v = std::isnan(dat[i]) ? 0.f : c_b[dat[i]];
        fwrite(&v, sizeof(float), 1, outf);
    }
    fclose(outf);
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 5: Count — pixel count for a one-hot binary component mask
// ═══════════════════════════════════════════════════════════════════════════
// For a one-hot mask we expect exactly one non-zero, non-NaN value.
// Returns the pixel count of that value, or -1 if conditions are not met.
// Crashes (exits) if the mask has more than one distinct non-zero label,
// since that would indicate an unexpected non-one-hot mask.

static long int stage_count_onehot(float *dat, size_t np) {
    map<float, size_t> count;
    size_t n_nan = 0;
    for (size_t i = 0; i < np; ++i) {
        float d = dat[i];
        if (std::isnan(d) || std::isinf(d)) { n_nan++; continue; }
        if (d == 0.f) continue;  // background
        count[d]++;
    }
    if (count.empty()) return 0;
    if (count.size() != 1) {
        fprintf(stderr,
            "stage_count_onehot: expected exactly 1 distinct non-zero label, "
            "found %zu. This mask is not one-hot. Aborting.\n",
            count.size());
        exit(1);
    }
    return (long int)count.begin()->second;
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 6: One-hot split + per-component output
// ═══════════════════════════════════════════════════════════════════════════
// Iterates over recoded labels. For each label >= 1:
//   - builds binary mask
//   - counts pixels
//   - if >= point_threshold: writes .bin/.hdr and emits "+component N count"
// Background (0) and label 1 (first/empty class from flood fill background)
// are skipped.

static void stage_onehot_output(float *recode, size_t nrow, size_t ncol,
                                 size_t n_classes,
                                 long int point_threshold,
                                 const str &base_fn,
                                 const str &base_hfn)
{
    size_t np = nrow * ncol;

    for (size_t N = 1; N <= n_classes; ++N) {
        float label = (float)N;

        // Build one-hot mask
        float *mask = falloc(np);
        for (size_t i = 0; i < np; ++i) {
            float d = recode[i];
            if (std::isnan(d)) {
                mask[i] = NAN;
            } else {
                mask[i] = (d == label) ? 1.f : 0.f;
            }
        }

        long int n_px = stage_count_onehot(mask, np);

        if (n_px < point_threshold) {
            printf("SKIP component %zu (%ld pixels, threshold %ld)\n",
                   N, n_px, point_threshold);
            free(mask);
            continue;
        }

        // Write component mask
        char idx_buf[8];
        snprintf(idx_buf, sizeof(idx_buf), "%03zu", N);
        str comp_fn  = base_fn + "_comp_" + idx_buf + ".bin";
        str comp_hfn = base_fn + "_comp_" + idx_buf + ".hdr";

        // Copy geometry from base header
        run("cp " + base_hfn + " " + comp_hfn);
        bwrite(mask, comp_fn, nrow, ncol, 1);

        // Emit parseable line for class_brush.py
        printf("+component %zu %ld\n", N, n_px);
        fflush(stdout);

        free(mask);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    if (argc < 4) {
        err("Usage: class_brush.exe <input_mask.bin> <brush_size> <point_threshold>");
    }

    str fn(argv[1]);
    long int brush_size      = atol(argv[2]);
    long int point_threshold = atol(argv[3]);

    if (brush_size <= 0)      err("brush_size must be > 0");
    if (point_threshold <= 0) err("point_threshold must be > 0");

    str hfn(hdr_fn(fn));
    size_t nrow, ncol, nband;
    size_t d_type = hread(hfn, nrow, ncol, nband);
    if (d_type != 4) err("Expected float32 (ENVI type 4) input image");
    if (nband  != 1) err("Expected single-band image");
    if (nrow   == 0 || ncol == 0) err("Image has zero rows or columns");
    if (mul_overflow(nrow, ncol)) err("nrow * ncol overflows size_t");

    size_t np = nrow * ncol;

    // ── Read input ──────────────────────────────────────────────────────────
    float *dat = bread(fn, nrow, ncol, nband);
    if (!dat) err("Failed to read input file");

    // ── Stage 1: Flood fill ─────────────────────────────────────────────────
    printf("Stage 1: flood fill...\n"); fflush(stdout);
    size_t n_flood_labels = 0;
    float *flood = stage_flood(dat, nrow, ncol, n_flood_labels);
    printf("  flood labels: %zu\n", n_flood_labels); fflush(stdout);

    // Write flood result
    str flood_fn  = fn + "_flood4.bin";
    str flood_hfn = fn + "_flood4.hdr";
    bwrite(flood, flood_fn, nrow, ncol, 1);
    hwrite(flood_hfn, nrow, ncol, 1, 4);

    free(dat);  // no longer needed

    // ── Stage 2: Link ───────────────────────────────────────────────────────
    printf("Stage 2: link (brush_size=%ld)...\n", brush_size); fflush(stdout);
    float *linked = stage_link(flood, nrow, ncol, brush_size);
    free(flood);

    str link_fn  = flood_fn + "_link.bin";
    str link_hfn = flood_fn + "_link.hdr";
    bwrite(linked, link_fn, nrow, ncol, 1);
    hwrite(link_hfn, nrow, ncol, 1, 4);

    // ── Stage 3: Recode ─────────────────────────────────────────────────────
    printf("Stage 3: recode...\n"); fflush(stdout);
    size_t n_classes = 0;
    float *recoded = stage_recode(linked, np, n_classes);
    free(linked);
    printf("  classes after recode: %zu\n", n_classes); fflush(stdout);

    str recode_fn  = link_fn + "_recode.bin";
    str recode_hfn = link_fn + "_recode.hdr";
    bwrite(recoded, recode_fn, nrow, ncol, 1);
    hwrite(recode_hfn, nrow, ncol, 1, 4);

    // ── Stage 4: Colour-wheel visualisation ─────────────────────────────────
    printf("Stage 4: colour wheel...\n"); fflush(stdout);
    str wheel_fn  = recode_fn + "_wheel.bin";
    str wheel_hfn = recode_fn + "_wheel.hdr";
    stage_wheel(recoded, nrow, ncol, n_classes, wheel_fn, wheel_hfn);

    // ── Stages 5 & 6: One-hot split, count, write per-component masks ───────
    printf("Stage 5/6: one-hot component output (threshold=%ld)...\n",
           point_threshold); fflush(stdout);
    stage_onehot_output(recoded, nrow, ncol, n_classes,
                        point_threshold, fn, str(hfn));

    free(recoded);
    printf("class_brush: done.\n");
    return 0;
}


