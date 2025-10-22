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
   (with progress monitor and multithreaded mode computation)
*/


/* 20251022: Segment-based central statistics and classification
   (parallelized within-segment mode computation using pthreads)
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
#include <thread>
#include <mutex>
#include <atomic>
#include <pthread.h>

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

static inline bool isnan_any(const std::vector<float> &v) {
  for (float f : v)
    if (std::isnan(f)) return true;
  return false;
}

struct ModeThreadContext {
  const std::vector<std::vector<float>> *pixlist;
  std::atomic<size_t> *job_idx;
  size_t total;
  std::mutex *best_mutex;
  double *best_sum;
  size_t *best_idx;
};

void *mode_inner_worker(void *arg) {
  ModeThreadContext *ctx = (ModeThreadContext *)arg;
  const auto &pixlist = *(ctx->pixlist);

  while (true) {
    size_t a = (*ctx->job_idx)++;
    if (a >= ctx->total) break;

    double s = 0.0;
    for (size_t b = 0; b < ctx->total; ++b) {
      if (a == b) continue;
      s += sqdist(pixlist[a], pixlist[b]);
    }

    {
      std::lock_guard<std::mutex> lock(*ctx->best_mutex);
      if (s < *ctx->best_sum) {
        *ctx->best_sum = s;
        *ctx->best_idx = a;
      }
    }
  }

  return nullptr;
}

static size_t find_mode_index_parallel(const std::vector<std::vector<float>> &pixlist) {
  if (pixlist.empty()) return 0;
  if (pixlist.size() == 1) return 0;

  unsigned nthreads = std::thread::hardware_concurrency();
  if (nthreads == 0) nthreads = 4;

  pthread_t *threads = new pthread_t[nthreads];
  ModeThreadContext ctx;
  ctx.pixlist = &pixlist;
  ctx.job_idx = new std::atomic<size_t>(0);
  ctx.total = pixlist.size();
  ctx.best_mutex = new std::mutex();
  ctx.best_sum = new double(std::numeric_limits<double>::infinity());
  ctx.best_idx = new size_t(0);

  for (unsigned t = 0; t < nthreads; ++t)
    pthread_create(&threads[t], nullptr, mode_inner_worker, &ctx);
  for (unsigned t = 0; t < nthreads; ++t)
    pthread_join(threads[t], nullptr);

  size_t result = *ctx.best_idx;

  delete ctx.job_idx;
  delete ctx.best_mutex;
  delete ctx.best_sum;
  delete ctx.best_idx;
  delete[] threads;
  return result;
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

  // Read headers
  str h_img(hdr_fn(img_fn));
  str h_seg(hdr_fn(seg_fn));
  str h_cls(hdr_fn(cls_fn));

  size_t nrow_img, ncol_img, nband_img;
  size_t nrow_seg, ncol_seg, nband_seg;
  size_t nrow_cls, ncol_cls, nband_cls;
  std::vector<std::string> bandNames;

  size_t dtype_img = hread(h_img, nrow_img, ncol_img, nband_img, bandNames);
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

  // Read data
  float *img = bread(img_fn, nrow, ncol, nband);
  float *seg = bread(seg_fn, nrow, ncol, nband_seg);
  float *cls = bread(cls_fn, nrow, ncol, nband_cls);

  if (!img || !seg || !cls)
    err("Failed to read one or more inputs.");

  // -------------------------------
  // Compute per-segment means
  // -------------------------------
  std::unordered_map<int, VecSum> seg_stats;
  std::cout << "Computing per-segment averages..." << std::endl;

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

  std::unordered_map<int, std::vector<float>> seg_means;
  seg_means.reserve(seg_stats.size());
  for (auto &kv : seg_stats) {
    if (kv.second.count == 0) continue;
    std::vector<float> mean(nband);
    for (size_t b = 0; b < nband; ++b)
      mean[b] = (float)(kv.second.sum[b] / kv.second.count);
    seg_means[kv.first] = std::move(mean);
  }

  // -------------------------------
  // MODE OPTION: parallel inner loop
  // -------------------------------
  if (use_mode) {
    std::cout << "Computing per-segment mode (parallelized inner loop)..." << std::endl;

    std::unordered_map<int, std::vector<std::vector<float>>> seg_pixels;
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

    size_t nseg = seg_pixels.size();
    size_t count = 0;
    for (auto &kv : seg_pixels) {
      count++;
      if (count % 50 == 0 || count == nseg)
        std::cout << "\rProcessed " << count << " / " << nseg << " segments" << std::flush;

      const auto &pixlist = kv.second;
      if (pixlist.empty()) continue;
      size_t best_idx = find_mode_index_parallel(pixlist);
      seg_means[kv.first] = pixlist[best_idx];
    }
    std::cout << "\rProcessed " << nseg << " / " << nseg << " segments\n";
  }

  // -------------------------------
  // Compute per-class averages of segment means
  // -------------------------------
  std::unordered_map<int, VecSum> class_accum;
  std::cout << "Computing per-class averages..." << std::endl;
  size_t np_step = np / 20;
  for (size_t i = 0; i < np; ++i) {
    if (i % np_step == 0)
      std::cout << "\rProgress " << (100 * i / np) << "%" << std::flush;

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
  std::cout << "\rProgress 100%\n";

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
  std::cout << "Writing segment-level image..." << std::endl;
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
  hwrite(out_seg_hdr, nrow, ncol, nband, 4, bandNames);

  // -------------------------------
  // Classify each pixel by nearest class representative
  // -------------------------------
  std::vector<float> out_class(np, NAN);
  std::cout << "Computing classification map..." << std::endl;
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
  hwrite(out_cls_hdr, nrow, ncol, 1, 4, bandNames);

  free(img);
  free(seg);
  free(cls);

  std::cout << "Done." << std::endl;
  return 0;
}

