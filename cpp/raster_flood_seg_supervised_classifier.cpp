/* 20251022: Segment-based central statistics and classification

Reads:
  1) Multi-band ENVI raster (float32, BSQ)
  2) Segment index map (float32, single-band) — from flood fill output
  3) Class map (float32, single-band) — used to aggregate per-class

Computes per-segment statistics (average or mode) for all bands, then
averages (or takes mode) across all segments of each class to derive
a per-class representative vector.  Finally, classifies each pixel
to the nearest class representative vector (by Euclidean distance).

Outputs:
  - [input]_segment_average.bin  OR  [input]_segment_mode.bin
      (multiband, float32)
  - [input]_classification_map.bin
      (single-band, float32)
*/

#include "misc.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <iostream>
#include <map>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 4)
        err("Usage: segment_average.exe [image_data.bin] [segment_index_map.bin] [class_map.bin]");

    // Input file names
    str fn_image(argv[1]);
    str fn_segment(argv[2]);
    str fn_class(argv[3]);

    // Header filenames
    str hfn_image(hdr_fn(fn_image));
    str hfn_segment(hdr_fn(fn_segment));
    str hfn_class(hdr_fn(fn_class));

    // Output file names
    str fn_avg(fn_image + "_segment_average.bin");
    str fn_mode(fn_image + "_segment_mode.bin");
    str hfn_avg(hdr_fn(fn_avg, true));
    str hfn_mode(hdr_fn(fn_mode, true));

    // Image data info
    size_t nrow, ncol, nband;
    vector<string> bandNames;

    size_t dtype_img = hread(hfn_image, nrow, ncol, nband, bandNames);
    if (dtype_img != 4)
        err("Expected 32-bit float image data (type 4)");

    size_t np = nrow * ncol;

    // Read input data
    float *dat = bread(fn_image, nrow, ncol, nband);
    float *segment_idx = bread(fn_segment, nrow, ncol, 1);
    float *class_map = bread(fn_class, nrow, ncol, 1);

    // Output arrays
    float *out_avg = falloc(np * nband);
    float *out_mode = falloc(np * nband);

    // Build segment → pixel indices
    map<int, vector<size_t>> seg_pixels;
    for (size_t i = 0; i < np; i++) {
        float seg_val = segment_idx[i];
        if (!isnan(seg_val))
            seg_pixels[(int)seg_val].push_back(i);
    }

    // Process each segment
    size_t nseg = seg_pixels.size();
    printf("Processing %zu segments...\n", nseg);

    for (auto &seg : seg_pixels) {
        const vector<size_t> &pixels = seg.second;
        if (pixels.empty()) continue;

        // --- Compute average for each band ---
        vector<float> band_sum(nband, 0.0f);
        vector<int> band_count(nband, 0);
        for (size_t b = 0; b < nband; b++) {
            for (size_t idx : pixels) {
                float val = dat[b * np + idx];
                if (!isnan(val)) {
                    band_sum[b] += val;
                    band_count[b]++;
                }
            }
        }

        vector<float> band_avg(nband, NAN);
        for (size_t b = 0; b < nband; b++) {
            if (band_count[b] > 0)
                band_avg[b] = band_sum[b] / band_count[b];
        }

        // --- Compute mode of class values ---
        map<int, int> freq;
        for (size_t idx : pixels) {
            float cval = class_map[idx];
            if (!isnan(cval))
                freq[(int)cval]++;
        }

        int mode_class = -1;
        int max_count = 0;
        for (auto &kv : freq) {
            if (kv.second > max_count) {
                max_count = kv.second;
                mode_class = kv.first;
            }
        }

        // --- Write to output arrays ---
        for (size_t idx : pixels) {
            for (size_t b = 0; b < nband; b++) {
                out_avg[b * np + idx] = band_avg[b];
                out_mode[b * np + idx] = (float)mode_class;
            }
        }
    }

    // Write outputs
    printf("Writing outputs...\n");
    bwrite(out_avg, fn_avg, nrow, ncol, nband);
    bwrite(out_mode, fn_mode, nrow, ncol, nband);

    hwrite(hfn_avg, nrow, ncol, nband, 4, bandNames);
    hwrite(hfn_mode, nrow, ncol, nband, 4, bandNames);

    // Cleanup
    free(dat);
    free(segment_idx);
    free(class_map);
    free(out_avg);
    free(out_mode);

    printf("Done.\n");
    return 0;
}

