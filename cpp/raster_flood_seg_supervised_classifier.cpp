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
   Fully parallelized L1-distance mode/non-mode computation
   Float32 operations, progress monitor, persistent thread pool
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
#include <queue>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>

using std::size_t;
using namespace std::chrono;

struct VecSum {
    std::vector<float> sum;
    long count = 0;
};

struct VecFloat {
    std::vector<float> v;
};

// L1 distance between two float vectors
static inline float l1_dist(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        d += std::fabs(a[i] - b[i]);
    return d;
}

// ThreadPool for persistent threads
class ThreadPool {
public:
    ThreadPool(size_t n_threads) : done(false) {
        for (size_t i = 0; i < n_threads; ++i)
            workers.emplace_back(&ThreadPool::worker_loop, this);
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            done = true;
        }
        cond.notify_all();
        for (auto &t : workers) t.join();
    }

    void enqueue(std::function<void()> f) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.push(std::move(f));
        }
        cond.notify_one();
    }

    void wait_for_completion() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            if (tasks.empty() && active.load() == 0) break;
            cond.wait_for(lock, std::chrono::milliseconds(50));
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable cond;
    std::atomic<int> active{0};
    bool done;

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cond.wait(lock, [this]() { return done || !tasks.empty(); });
                if (done && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
                active++;
            }
            task();
            active--;
            cond.notify_all();
        }
    }
};

int main(int argc, char **argv) {
    if (argc < 4) {
        err("Usage: raster_flood_seg_supervised_classifier [image.bin] [segment_index.bin] [class_map.bin] [--mode]");
    }

    bool use_mode = false;
    if (argc > 4 && strcmp(argv[4], "--mode") == 0)
        use_mode = true;

    std::string img_fn(argv[1]);
    std::string seg_fn(argv[2]);
    std::string cls_fn(argv[3]);

    // Read headers
    std::string h_img = hdr_fn(img_fn);
    std::string h_seg = hdr_fn(seg_fn);
    std::string h_cls = hdr_fn(cls_fn);

    size_t nrow_img, ncol_img, nband_img;
    size_t nrow_seg, ncol_seg, nband_seg;
    size_t nrow_cls, ncol_cls, nband_cls;
    std::vector<std::string> bandNames;

    hread(h_img, nrow_img, ncol_img, nband_img, bandNames);
    hread(h_seg, nrow_seg, ncol_seg, nband_seg, bandNames);
    hread(h_cls, nrow_cls, ncol_cls, nband_cls, bandNames);

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
    // Compute per-segment means or collect pixel vectors
    // -------------------------------
    std::unordered_map<int, VecSum> seg_stats;
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

        if (use_mode) {
            seg_pixels[seg_id].push_back(pix);
        } else {
            VecSum &vs = seg_stats[seg_id];
            if (vs.sum.empty()) vs.sum.assign(nband, 0.f);
            for (size_t b = 0; b < nband; ++b) vs.sum[b] += pix[b];
            vs.count++;
        }
    }

    // Compute per-segment representatives
    std::unordered_map<int, std::vector<float>> seg_means;

    if (!use_mode) {
        for (auto &kv : seg_stats) {
            if (kv.second.count == 0) continue;
            std::vector<float> mean(nband);
            for (size_t b = 0; b < nband; ++b)
                mean[b] = kv.second.sum[b] / kv.second.count;
            seg_means[kv.first] = mean;
        }
    } else {
        // Mode: persistent thread pool with inner L1 parallelization
        std::vector<int> seg_list;
        for (auto &kv : seg_pixels) seg_list.push_back(kv.first);

        std::atomic<size_t> next_seg{0};
        ThreadPool pool(2 * std::thread::hardware_concurrency());

        std::mutex seg_means_mutex;
        std::atomic<size_t> jobs_done{0};
        size_t total_jobs = 0;
        for (auto &kv : seg_pixels)
            total_jobs += kv.second.size();

        auto start_time = steady_clock::now();

        for (size_t t = 0; t < seg_list.size(); ++t) {
            pool.enqueue([&, t]() {
                int sid = seg_list[t];
                auto &pixels = seg_pixels[sid];
                size_t npix = pixels.size();
                std::vector<float> best_vec(nband);
                float best_sum = std::numeric_limits<float>::infinity();

                for (size_t i = 0; i < npix; ++i) {
                    float s = 0.f;
                    for (size_t j = 0; j < npix; ++j) {
                        if (i == j) continue;
                        s += l1_dist(pixels[i], pixels[j]);
                    }
                    if (s < best_sum) {
                        best_sum = s;
                        best_vec = pixels[i];
                    }

                    // Progress monitor (once every 1000 pixels)
                    size_t done_now = jobs_done.fetch_add(1) + 1;
                    if (done_now % 1000 == 0) {
                        double frac = (double)done_now / total_jobs;
                        auto now = steady_clock::now();
                        double elapsed = duration_cast<seconds>(now - start_time).count();
                        double eta = elapsed / frac - elapsed;
                        printf("Mode progress: %zu / %zu (%.2f%%), ETA %.0f s\n",
                               done_now, total_jobs, frac * 100., eta);
                    }
                }

                std::lock_guard<std::mutex> lock(seg_means_mutex);
                seg_means[sid] = best_vec;
            });
        }

        pool.wait_for_completion();
    }

    // -------------------------------
    // Compute per-class averages
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
        if (vc.sum.empty()) vc.sum.assign(nband, 0.f);
        for (size_t b = 0; b < nband; ++b)
            vc.sum[b] += it->second[b];
        vc.count++;
    }

    std::unordered_map<int, std::vector<float>> class_means;
    for (auto &kv : class_accum) {
        if (kv.second.count == 0) continue;
        std::vector<float> mean(nband);
        for (size_t b = 0; b < nband; ++b)
            mean[b] = kv.second.sum[b] / kv.second.count;
        class_means[kv.first] = mean;
    }

    // -------------------------------
    // Build output segment image
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

    std::string out_seg_fn = img_fn + (use_mode ? "_segment_mode.bin" : "_segment_average.bin");
    std::string out_seg_hdr = hdr_fn(out_seg_fn, true);
    bwrite(out_segment.data(), out_seg_fn, nrow, ncol, nband);
    hwrite(out_seg_hdr, nrow, ncol, nband, 4, bandNames);

    // -------------------------------
    // Classify pixels
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

        float best_d = std::numeric_limits<float>::infinity();
        int best_cls = -9999;
        for (auto &kv : class_means) {
            float d = l1_dist(pix, kv.second);
            if (d < best_d) {
                best_d = d;
                best_cls = kv.first;
            }
        }
        out_class[i] = best_cls;
    }

    std::string out_cls_fn = img_fn + "_classification_map.bin";
    std::string out_cls_hdr = hdr_fn(out_cls_fn, true);
    bwrite(out_class.data(), out_cls_fn, nrow, ncol, 1);
    hwrite(out_cls_hdr, nrow, ncol, 1, 4);

    free(img);
    free(seg);
    free(cls);

    return 0;
}

