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


/* 20251022: Segment-based central statistics and classification

Reads:
  1) Multi-band ENVI raster (float32, BSQ)  <-- band names read and propagated
  2) Segment index map (float32, single-band) — from flood fill output
  3) Class map (float32, single-band) — used to aggregate per-class

Computes per-segment statistics (average or mode) for all bands, then
averages (or takes mode) across all segments of each class to derive
a per-class representative vector.  Finally, classifies each pixel
to the nearest class representative vector (by Euclidean distance).

Outputs:
  - [input]_segment_average.bin  OR  [input]_segment_mode.bin
      (multiband, float32) — header will contain original band names
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
#include <string>

using std::size_t;

struct VecSum {
  std::vector<double> sum;
  long count = 0;
};

static inline double sqdist(const std::vector<float> &a, const std::vector<float> &b) {
  double d = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = (double)a[i] - (double)b[i];
    d += diff * diff;
  }
  return d;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    err("Usage: segment_central_stats.exe [image.bin] [segment_index.bin] [class_map.bin] [--mode (optional)]");
  }

  bool use_mode = false;
  if (argc > 4 && strcmp(argv[4], "--mode") == 0) {
    use_mode = true;
  }

  str img_fn(argv[1]);
  str seg_fn(argv[2]);
  str cls_fn(argv[3]);

  // Read headers (image header read with band names)
  str h_img(hdr_fn(img_fn));
  str h_seg(hdr_fn(seg_fn));
  str h_cls(hdr_fn(cls_fn));

  size_t nrow_img=0, ncol_img=0, nband_img=0;
  size_t nrow_seg=0, ncol_seg=0, nband_seg=0;
  size_t nrow_cls=0, ncol_cls=0, nband_cls=0;

  // NEW: read band names from image header
  std::vector<std::string> bandNames;
  size_t dtype_img = hread(h_img, nrow_img, ncol_img, nband_img, bandNames);

  // read segment and class headers with existing signature (no band names needed)
  size_t dtype_seg = hread(h_seg, nrow_seg, ncol_seg, nband_seg);
  size_t dtype_cls = hread(h_cls, nrow_cls, ncol_cls, nband_cls);

  if (dtype_img != 4 || dtype_seg != 4 || dtype_cls != 4)
    err("All inputs must be 32-bit float (type 4).");

  if (nrow_img != nrow_seg || ncol_img != ncol_seg ||
      nrow_img != nrow_cls || ncol_img != ncol_cls)
    err("All input rasters must have the same dimensions.");

  if (nband_seg != 1 || nband_cls != 1)
    err("Segment index and class map must be single-band.");

  size_t nrow = nrow_img;
  size_t ncol = ncol_img;
  size_t nband = nband_img;
  size_t np = nrow * ncol;

  // Read data (all float32 BSQ)
  float *img = bread(img_fn, nrow, ncol, nband);
  float *seg = bread(seg_fn, nrow, ncol, nband_seg);
  float *cls = bread(cls_fn, nrow, ncol, nband_cls);

  if (!img || !seg || !cls)
    err("Failed to read one or more inputs.");

  // -------------------------------
  // Compute per-segment means
  // -------------------------------
  std::unordered_map<int, VecSum> seg_stats;

  for (size_t i = 0; i < np; ++i) {
    float seg_id_f = seg[i];
    if (std::isnan(seg_id_f)) continue;
    int seg_id = (int)seg_id_f;

    std::vector<float> pix(nband);
    bool skip = false;
    for (size_t b = 0; b < nband; ++b) {
      float v = img[b * np + i];
      if (std::isnan(v)) { skip = true; break; }
      pix[b] = v;
    }
    if (skip) continue;

    VecSum &vs = seg_stats[seg_id];
    if (vs.sum.empty()) vs.sum.assign(nband, 0.0);
    for (size_t b = 0; b < nband; ++b)
      vs.sum[b] += pix[b];
    vs.count++;
  }

  // Convert to per-segment means
  std::unordered_map<int, std::vector<float>> seg_means;
  for (auto &kv : seg_stats) {
    if (kv.second.count == 0) continue;
    std::vector<float> mean(nband);
    for (size_t b = 0; b < nband; ++b)
      mean[b] = (float)(kv.second.sum[b] / kv.second.count);
    seg_means[kv.first] = std::move(mean);
  }

  // If using mode instead of mean, recompute per-segment representative as mode
  if (use_mode) {
    std::unordered_map<int, std::vector<std::vector<float>>> seg_pixels;

    // Collect per-segment pixel vectors
    for (size_t i = 0; i < np; ++i) {
      float seg_id_f = seg[i];
      if (std::isnan(seg_id_f)) continue;
      int seg_id = (int)seg_id_f;

      std::vector<float> pix(nband);
      bool skip = false;
      for (size_t b = 0; b < nband; ++b) {
        float v = img[b * np + i];
        if (std::isnan(v)) { skip = true; break; }
        pix[b] = v;
      }
      if (skip) continue;

      seg_pixels[seg_id].push_back(std::move(pix));
    }

    // Compute per-segment "mode" = vector with min total distance to others
    for (auto &kv : seg_pixels) {
      int sid = kv.first;
      auto &pixlist = kv.second;
      if (pixlist.empty()) continue;

      double best_sum = std::numeric_limits<double>::infinity();
      size_t best_idx = 0;
      for (size_t a = 0; a < pixlist.size(); ++a) {
        double s = 0.0;
        for (size_t b = 0; b < pixlist.size(); ++b) {
          if (a == b) continue;
          s += sqdist(pixlist[a], pixlist[b]);
        }
        if (s < best_sum) {
          best_sum = s;
          best_idx = a;
        }
      }
      seg_means[sid] = pixlist[best_idx];
    }
  }

  // -------------------------------
  // Compute per-class averages of segment means
  // -------------------------------
  std::unordered_map<int, VecSum> class_accum;
  for (size_t i = 0; i < np; ++i) {
    float seg_id_f = seg[i];
    float cls_id_f = cls[i];
    if (std::isnan(seg_id_f) || std::isnan(cls_id_f)) continue;
    int seg_id = (int)seg_id_f;
    int cls_id = (int)cls_id_f;
    auto it = seg_means.find(seg_id);
    if (it == seg_means.end()) continue;

    VecSum &vc = class_accum[cls_id];
    if (vc.sum.empty()) vc.sum.assign(nband, 0.0);
    for (size_t b = 0; b < nband; ++b)
      vc.sum[b] += it->second[b];
    vc.count++;
  }

  std::unordered_map<int, std::vector<float>> class_means;
  for (auto &kv : class_accum) {
    if (kv.second.count == 0) continue;
    std::vector<float> mean(nband);
    for (size_t b = 0; b < nband; ++b)
      mean[b] = (float)(kv.second.sum[b] / kv.second.count);
    class_means[kv.first] = std::move(mean);
  }

  // -------------------------------
  // Build output segment-average/mode image
  // -------------------------------
  std::vector<float> out_segment(nband * np, NAN);
  for (size_t i = 0; i < np; ++i) {
    float seg_id_f = seg[i];
    if (std::isnan(seg_id_f)) continue;
    int seg_id = (int)seg_id_f;
    auto it = seg_means.find(seg_id);
    if (it == seg_means.end()) continue;
    for (size_t b = 0; b < nband; ++b)
      out_segment[b * np + i] = it->second[b];
  }

  str out_seg_fn(img_fn + (use_mode ? "_segment_mode.bin" : "_segment_average.bin"));
  str out_seg_hdr(hdr_fn(out_seg_fn, true));
  bwrite(out_segment.data(), out_seg_fn, nrow, ncol, nband);

  // NEW: write band names into the segment-average/mode header
  hwrite(out_seg_hdr, nrow, ncol, nband, 4, bandNames);

  // -------------------------------
  // Classify each pixel by nearest class representative
  // -------------------------------
  std::vector<float> out_class(np, NAN);
  for (size_t i = 0; i < np; ++i) {
    std::vector<float> pix(nband);
    bool skip = false;
    for (size_t b = 0; b < nband; ++b) {
      float v = img[b * np + i];
      if (std::isnan(v)) { skip = true; break; }
      pix[b] = v;
    }
    if (skip) continue;

    double best_d = std::numeric_limits<double>::infinity();
    int best_cls = -9999;
    for (auto &kv : class_means) {
      double d = sqdist(pix, kv.second);
      if (d < best_d) {
        best_d = d;
        best_cls = kv.first;
      }
    }
    out_class[i] = (float)best_cls;
  }

  str out_cls_fn(img_fn + "_classification_map.bin");
  str out_cls_hdr(hdr_fn(out_cls_fn, true));
  bwrite(out_class.data(), out_cls_fn, nrow, ncol, 1);
  hwrite(out_cls_hdr, nrow, ncol, 1, 4);

  // -------------------------------
  // Cleanup
  // -------------------------------
  free(img);
  free(seg);
  free(cls);

  return 0;
}


