/* ===========================================================================
 * kgc.cpp -- k-nearest-neighbour density clustering with hint-guided
 *            selection of the neighbourhood size.
 *
 * Pipeline
 *   1. read an ENVI raster (bip, bsq or bil) and convert to bip in RAM
 *   2. precondition: scale every band to [0,1] by (x - min) / (max - min)
 *   3. setoid deduplication of the whole image: pixels whose band vectors are
 *      exactly equal form one class, one representative is retained
 *   4. decimate the deduplicated set by a regular stride n_skip
 *   5. neighbour table: the k_max nearest OTHER points of each retained point
 *   6. for K = K_min, K_min+1, ... : density rho_K(i) = -S_K(i), hill climb to
 *      the densest point of the K-neighbourhood, iterate to a fixed point.
 *      Each K gives its own clustering, computed fresh.
 *   7. score every class of every K against the hint and keep the best; report
 *      that K together with the best class at K-1 and K+1
 *
 * Steps 3 and 5 are memoised on disk against a checksum of the image binary.
 *
 * Build:  g++ -O3 -std=c++11 -pthread kgc.cpp -o kgc
 * Usage:  ./kgc --help
 * ===========================================================================*/
#include "kgc.h"

using namespace kgc;

static void usage(const char* prog) {
  std::printf(
"\n"
"kgc -- kNN density clustering with hint-guided choice of K\n"
"\n"
"USAGE\n"
"  %s <image.bin> --hint <hint.bin> [options]\n"
"\n"
"INPUT\n"
"  <image.bin>            ENVI raster, data type 4 (32-bit float), any number\n"
"                         of bands, interleave bip, bsq or bil.  The matching\n"
"                         .hdr is found automatically.\n"
"  --hint <hint.bin>      1-band ENVI type-4 mask, 1.0 true and 0.0 false,\n"
"                         matching the image in samples and lines.\n"
"\n"
"PARAMETERS\n"
"  --nskip <s>            Decimation stride over the deduplicated set: every\n"
"                         s-th record is retained.  Default: chosen so that\n"
"                         about --budget-points points remain.\n"
"  --kmax <k>             Neighbours held per point, which bounds the memory\n"
"                         of the neighbour table and the largest K the sweep\n"
"                         can reach.  Default 10000.  The table is\n"
"                         n_points x k_max entries of 12 bytes, so this is the\n"
"                         parameter to lower if memory is tight.\n"
"  --kstep <s>            The sweep visits K = s, 2s, 3s, ... up to --kmax, so\n"
"                         the first K is also the step.  Default 5.\n"
"  --patience <n>         Stop the sweep once this many completed levels have\n"
"                         gone by above the best K without improving on it.\n"
"                         Default 0, meaning off: the score against the hint is\n"
"                         not unimodal in K, so stopping on a local plateau can\n"
"                         miss a much better class at larger K.  Measured on\n"
"                         C50929, a patience of 40 stopped at K=535 with mutual\n"
"                         information 0.247 and missed K=875 at 0.451.\n"
"  --min-class <n>        Smallest class eligible for selection.\n"
"                         Default sqrt(retained points).\n"
"  --budget-points <n>    Target retained-point count driving --nskip (20000).\n"
"  --threads <n>          Worker threads.  Default: hardware concurrency.\n"
"  --no-cache             Recompute the deduplication and neighbour table even\n"
"                         if a valid cache is present.\n"
"\n"
"OUTPUT\n"
"  --out <prefix>         Output prefix.  Default: the image path minus .bin.\n"
"\n"
"  Every raster written is ENVI data type 4, BSQ interleave, the same size as\n"
"  the input, carrying the input's map info, projection info and coordinate\n"
"  system string when present, with descriptive band names.\n"
"\n"
"  <prefix>_selected.bin  6 bands: the class chosen at the best K, the hint,\n"
"                         the codeword log-likelihood ratio, the class map at\n"
"                         the best K, and the best class at K-1 and at K+1\n"
"  <prefix>_klevels.csv   one row per K: classes, best class and its scores\n"
"  <prefix>_params.txt    parameters and the chosen K\n"
"  <image>.kgc_dedup      memoised deduplication, keyed on a checksum of the\n"
"                         image binary\n"
"  <image>.kgc_knn_*      memoised neighbour table, same key plus n_skip and\n"
"                         k_max\n"
"\n"
"EXAMPLE\n"
"  %s scene.bin --hint perimeter.bin --out run1\n"
"\n", prog, prog);
}

struct Args {
  str image, hint, out;
  long nskip = -1, kmax = 10000, kstep = 5, min_class = -1, patience = 0;
  long budget_points = 20000, threads = -1;
  bool no_cache = false;
};

/* ------------------------------------------------------------------------ */
/* neighbour table                                                           */
/* ------------------------------------------------------------------------ */

/* The k_max nearest OTHER points of every retained point, sorted ascending.
 * K counts real neighbours: the point itself is never included.
 *
 * Brute force with a partial selection, because k_max is a sizeable fraction
 * of the point count and a tree degenerates to a full scan at that ratio
 * while paying for the traversal as well.                                  */
/* The neighbour table is the whole cost of this program: O(n^2 * dim)
 * distances, and the table itself is n * kmax * 12 bytes. It is
 * therefore the only part worth moving to a GPU.
 *
 * kgc.cu compiles THIS FILE with KGC_BUILD_KNN_EXTERNAL defined and
 * supplies a CUDA build_knn with the same signature. Everything else --
 * the dedup, the K sweep, the class selection, the output and every log
 * line -- is literally this code, so the two builds cannot drift apart
 * in behaviour. */
#ifdef KGC_BUILD_KNN_EXTERNAL
void build_knn_cuda(const float* pts, size_t n, size_t dim, size_t kmax,
                    size_t threads, std::vector<float>& dd,
                    std::vector<size_t>& di);
static void build_knn(const float* pts, size_t n, size_t dim, size_t kmax,
                      size_t threads, std::vector<float>& dd,
                      std::vector<size_t>& di) {
  build_knn_cuda(pts, n, dim, kmax, threads, dd, di);
}
#else
static void build_knn(const float* pts, size_t n, size_t dim, size_t kmax,
                      size_t threads, std::vector<float>& dd,
                      std::vector<size_t>& di) {
  dd.assign(n * kmax, 0.f);
  di.assign(n * kmax, 0);
  /* one job per row of the neighbour table */
  std::atomic<size_t> done(0);
  parallel_jobs(n, threads, [&](size_t q, size_t) {
    std::vector<std::pair<float, size_t> > v;
    v.reserve(n - 1);
    const float* a = pts + q * dim;
    for (size_t i = 0; i < n; i++) {
      if (i == q) continue; /* self excluded */
      const float* b = pts + i * dim;
      float s = 0.f;
      for (size_t m = 0; m < dim; m++) { float t = a[m] - b[m]; s += t * t; }
      v.push_back(std::make_pair(s, i));
    }
    size_t k = std::min(kmax, v.size());
    std::partial_sort(v.begin(), v.begin() + k, v.end());
    for (size_t m = 0; m < k; m++) {
      dd[q * kmax + m] = std::sqrt(v[m].first);
      di[q * kmax + m] = v[m].second;
    }
    for (size_t m = k; m < kmax; m++) {
      dd[q * kmax + m] = k ? dd[q * kmax + k - 1] : 0.f;
      di[q * kmax + m] = k ? di[q * kmax + k - 1] : q;
    }
    size_t d = ++done;
    if ((d % 2000) == 0)
      std::fprintf(stderr, "  neighbours %5.1f%%\r", 100.0 * d / n);
  });
  std::fprintf(stderr, "  neighbours 100.0%%\n");
}
#endif  /* KGC_BUILD_KNN_EXTERNAL */

/* ------------------------------------------------------------------------ */
/* on-disk memoisation                                                       */
/* ------------------------------------------------------------------------ */

static const uint64_t CACHE_MAGIC = 0x6b67635f76340000ull;

static bool cache_read_head(FILE* f, uint64_t want_sum, uint64_t want_tag) {
  uint64_t magic = 0, sum = 0, tag = 0;
  if (std::fread(&magic, sizeof magic, 1, f) != 1) return false;
  if (std::fread(&sum, sizeof sum, 1, f) != 1) return false;
  if (std::fread(&tag, sizeof tag, 1, f) != 1) return false;
  return magic == CACHE_MAGIC && sum == want_sum && tag == want_tag;
}

static void cache_write_head(FILE* f, uint64_t sum, uint64_t tag) {
  std::fwrite(&CACHE_MAGIC, sizeof CACHE_MAGIC, 1, f);
  std::fwrite(&sum, sizeof sum, 1, f);
  std::fwrite(&tag, sizeof tag, 1, f);
}

/* ------------------------------------------------------------------------ */
/* main                                                                      */
/* ------------------------------------------------------------------------ */

/* ------------------------------------------------------------------------ */
/* state shared with the worker threads                                      */
/* ------------------------------------------------------------------------ */

/* Read-only for the duration of the sweep.  These are file-scope so that every
 * worker reads one definite object rather than a captured reference whose
 * lifetime depends on the caller's frame.                                   */
struct Level {
  size_t K, n_classes;
  long best;
  double npix, nh, mi, lift;
  bool done;
  Level() : K(0), n_classes(0), best(-1), npix(0), nh(0), mi(0), lift(0), done(false) {}
};
struct Scratch { std::vector<double> S; std::vector<size_t> asc, peak; };

static size_t g_n = 0, g_nb = 0, g_kmax = 0, g_np = 0, g_min_class = 0;
static size_t g_kstep = 1, g_n_levels = 0, g_patience = 0;
static double g_HN = 0;
static const float*  g_dd = 0;   /* neighbour distances, g_n x g_kmax        */
static const size_t* g_di = 0;   /* neighbour indices,   g_n x g_kmax        */
static const double* g_w  = 0;   /* pixels represented by each point         */
static const double* g_h  = 0;   /* of those, hint pixels                    */
static std::vector<Scratch> g_scratch;   /* one per worker, never shared     */
static std::vector<Level>   g_levels;    /* one slot per level, never shared */

/* Written by the workers, and only these.  Guarded by one mutex; nothing else
 * needs a guard because nothing else changes while the sweep runs.          */
static pthread_mutex_t g_res_mtx;
static double g_best_mi = -1e300;
static long   g_best_K = -1;
static size_t g_since_best = 0, g_n_done = 0;
static volatile int g_stop = 0;

/* The clustering at one K, on this worker's own scratch.  Reads globals only.
 * If cls_out is given it also returns the class index of every point.       */
static Level compute_level(size_t K, Scratch& sc, std::vector<long>* cls_out) {
  const size_t n = g_n, kmax = g_kmax;
  std::vector<double>& S = sc.S;
  for (size_t i = 0; i < n; i++) {
    double s = 0;
    const float* d = &g_dd[i * kmax];
    for (size_t m = 0; m < K; m++) s += d[m];
    S[i] = s;
  }
  for (size_t i = 0; i < n; i++) {
    double br = -S[i];
    size_t bi = i;
    const size_t* ni = &g_di[i * kmax];
    for (size_t m = 0; m < K; m++) {
      size_t j = ni[m];
      if (-S[j] > br) { br = -S[j]; bi = j; }
    }
    sc.asc[i] = bi;
  }
  for (size_t i = 0; i < n; i++) {
    size_t j = i, g = 0;
    while (sc.asc[j] != j && g++ < n) j = sc.asc[j];
    sc.peak[i] = j;
  }
  std::map<size_t, size_t> id;
  std::vector<double> cw2, ch;
  std::vector<size_t> cn;
  if (cls_out) cls_out->assign(n, -1);
  for (size_t i = 0; i < n; i++) {
    std::map<size_t, size_t>::iterator it = id.find(sc.peak[i]);
    size_t c;
    if (it == id.end()) {
      c = cw2.size();
      id[sc.peak[i]] = c;
      cw2.push_back(0.0); ch.push_back(0.0); cn.push_back(0);
    } else c = it->second;
    cw2[c] += g_w[i]; ch[c] += g_h[i]; cn[c] += 1;
    if (cls_out) (*cls_out)[i] = (long)c;
  }
  Level L;
  L.K = K; L.n_classes = cw2.size(); L.done = true;
  double base = g_HN / (double)g_np, best = -1e300;
  for (size_t c = 0; c < cw2.size(); c++) {
    if (cn[c] < g_min_class || cw2[c] < 1.0) continue;
    double a = ch[c], b = cw2[c] - a;
    double lift = base > 0 ? (a / cw2[c]) / base : 0.0;
    if (!(lift > 1.0)) continue;
    double mi = mutual_information_2x2(a, b, g_HN - a, (double)g_np - g_HN - b);
    if (mi > best) {
      best = mi;
      L.best = (long)c; L.npix = cw2[c]; L.nh = a; L.mi = mi; L.lift = lift;
    }
  }
  return L;
}

/* One job is one level of K.  The clustering runs entirely on this worker's
 * own scratch; only the publish at the end touches shared state.           */
static void sweep_job(size_t job, size_t worker) {
  size_t K = (job + 1) * g_kstep;
  if (K > g_kmax) K = g_kmax;
  Level L = compute_level(K, g_scratch[worker], 0);

  pthread_mutex_lock(&g_res_mtx);
  g_levels[job] = L;
  g_n_done++;
  /* Deterministic tie-break: strictly better wins, an exact tie goes to the
   * smaller K.  Without it the winner among equally scoring levels would
   * depend on which worker published first.                                */
  const double tie = 1e-12;
  bool take = false;
  if (L.best >= 0) {
    if (g_best_K < 0) take = true;
    else if (L.mi > g_best_mi + tie) take = true;
    else if (L.mi > g_best_mi - tie && (long)K < g_best_K) take = true;
  }
  if (take) { g_best_mi = L.mi; g_best_K = (long)K; g_since_best = 0; }
  else if ((long)K > g_best_K) g_since_best++;

  if (L.n_classes <= 1) g_stop = 1;
  if (g_patience && g_since_best >= g_patience) g_stop = 1;

  if ((g_n_done % 50) == 0 || g_n_done == g_n_levels) {
    if (g_best_K > 0)
      std::fprintf(stderr, "  %zu/%zu levels, best K=%ld MI=%.4f\n",
                   g_n_done, g_n_levels, g_best_K, g_best_mi);
    else
      std::fprintf(stderr, "  %zu/%zu levels, no eligible class yet\n",
                   g_n_done, g_n_levels);
  }
  pthread_mutex_unlock(&g_res_mtx);
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(argv[0]); return 1; }
  Args A;
  for (int i = 1; i < argc; i++) {
    str a = argv[i];
    auto need = [&](const char* w) -> str {
      if (i + 1 >= argc) die(str("missing value for ") + w);
      return str(argv[++i]);
    };
    if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else if (a == "--hint") A.hint = need("--hint");
    else if (a == "--out") A.out = need("--out");
    else if (a == "--nskip") A.nskip = atol(need("--nskip").c_str());
    else if (a == "--kmax") A.kmax = atol(need("--kmax").c_str());
    else if (a == "--kstep") A.kstep = atol(need("--kstep").c_str());
    else if (a == "--patience") A.patience = atol(need("--patience").c_str());
    else if (a == "--min-class") A.min_class = atol(need("--min-class").c_str());
    else if (a == "--budget-points") A.budget_points = atol(need("--budget-points").c_str());
    else if (a == "--threads") A.threads = atol(need("--threads").c_str());
    else if (a == "--no-cache") A.no_cache = true;
    else if (!a.empty() && a[0] == '-') die("unknown option: " + a);
    else if (A.image.empty()) A.image = a;
    else die("unexpected argument: " + a);
  }
  if (A.image.empty()) { usage(argv[0]); return 1; }
  if (A.hint.empty()) die("a --hint mask is required (see --help)");
  size_t threads = A.threads > 0 ? (size_t)A.threads : cpu_count();
  std::fprintf(stderr, "[threads] %zu worker(s)\n", threads);
  str out = A.out.empty() ? strip_ext(A.image) : A.out;

  /* ---- 1. read ---------------------------------------------------- */
  EnviHeader H;
  float* img = read_envi_bip(A.image, H);
  size_t np = H.npix(), nb = H.bands;
  std::fprintf(stderr, "[input] %s: %zu x %zu x %zu, interleave %s -> bip\n",
               A.image.c_str(), H.samples, H.lines, nb, H.interleave.c_str());

  std::vector<float> hint;
  {
    EnviHeader HH;
    float* h = read_envi_bip(A.hint, HH);
    if (HH.samples != H.samples || HH.lines != H.lines)
      die("hint dimensions do not match the image");
    if (HH.bands != 1) die("hint must be a single band");
    hint.assign(h, h + np);
    std::free(h);
    for (std::map<str, str>::iterator it = HH.geo.begin(); it != HH.geo.end(); ++it)
      if (H.geo.find(it->first) == H.geo.end()) H.geo[it->first] = it->second;
  }

  /* pixels carrying a non-finite value form one equivalence class which takes
   * no part in the clustering; their pixels carry the reserved label -1     */
  std::vector<char> nodata(np, 0);
  size_t n_nodata = 0;
  for (size_t p = 0; p < np; p++) {
    for (size_t b = 0; b < nb; b++)
      if (!std::isfinite(img[p * nb + b])) { nodata[p] = 1; break; }
    if (nodata[p]) n_nodata++;
  }
  if (n_nodata)
    std::fprintf(stderr, "[input] %zu pixels carry a non-finite value; one setoid "
                         "class, excluded from the clustering\n", n_nodata);

  /* ---- 2. precondition: every band to [0,1] ----------------------- */
  {
    size_t nblk = (np + 65535) / 65536;
    std::vector<double> pmn(nblk * nb, 1e300), pmx(nblk * nb, -1e300);
    parallel_jobs(nblk, threads, [&](size_t blk, size_t) {
      size_t lo = blk * 65536, hi = std::min(lo + 65536, np);
      for (size_t p = lo; p < hi; p++) {
        if (nodata[p]) continue;
        for (size_t b = 0; b < nb; b++) {
          double v = img[p * nb + b];
          if (v < pmn[blk * nb + b]) pmn[blk * nb + b] = v;
          if (v > pmx[blk * nb + b]) pmx[blk * nb + b] = v;
        }
      }
    });
    std::vector<double> lo(nb, 1e300), hi(nb, -1e300);
    for (size_t blk = 0; blk < nblk; blk++)
      for (size_t b = 0; b < nb; b++) {
        if (pmn[blk * nb + b] < lo[b]) lo[b] = pmn[blk * nb + b];
        if (pmx[blk * nb + b] > hi[b]) hi[b] = pmx[blk * nb + b];
      }
    for (size_t b = 0; b < nb; b++) if (!(hi[b] > lo[b])) hi[b] = lo[b] + 1.0;
    parallel_for(np, threads, [&](size_t p, size_t) {
      if (nodata[p]) return;
      for (size_t b = 0; b < nb; b++)
        img[p * nb + b] = (float)((img[p * nb + b] - lo[b]) / (hi[b] - lo[b]));
    }, 4096);
    std::fprintf(stderr, "[precondition] each band scaled to [0,1] by (x-min)/(max-min)\n");
  }

  /* ---- 3. setoid deduplication of the whole image ----------------- */
  uint64_t sum = file_checksum(A.image);
  std::vector<size_t> ddup_rep;     /* representative pixel of each class     */
  std::vector<size_t> ddup_lookup(np, 0); /* pixel -> class index, npos if nodata */
  const size_t NODATA_CLASS = (size_t)-1;
  str dedup_cache = A.image + ".kgc_dedup";
  bool loaded = false;
  if (!A.no_cache && file_exists(dedup_cache)) {
    FILE* f = std::fopen(dedup_cache.c_str(), "rb");
    if (f) {
      if (cache_read_head(f, sum, (uint64_t)np * 1000003ull + nb)) {
        size_t nd = 0;
        if (std::fread(&nd, sizeof nd, 1, f) == 1) {
          ddup_rep.resize(nd);
          if (std::fread(&ddup_rep[0], sizeof(size_t), nd, f) == nd &&
              std::fread(&ddup_lookup[0], sizeof(size_t), np, f) == np)
            loaded = true;
        }
      }
      std::fclose(f);
    }
    if (loaded)
      std::fprintf(stderr, "[dedup] reusing %s (%zu classes)\n",
                   dedup_cache.c_str(), ddup_rep.size());
  }
  if (!loaded) {
    /* Exact grouping, parallelised: every pixel's band vector is hashed by a
     * worker, the pairs are sorted by hash, and equal-hash runs are then
     * compared byte for byte so that the grouping stays exact.  This replaces
     * an ordered map keyed on the raw bytes, which was the slowest serial step
     * in the pipeline.                                                      */
    std::vector<std::pair<uint64_t, size_t> > hp;
    hp.resize(np);
    parallel_for(np, threads, [&](size_t p, size_t) {
      uint64_t h = 1469598103934665603ull;
      const unsigned char* q = (const unsigned char*)(img + p * nb);
      for (size_t z = 0; z < nb * sizeof(float); z++) { h ^= q[z]; h *= 1099511628211ull; }
      hp[p] = std::make_pair(h, p);
    }, 4096);
    /* no-data pixels are one class of their own and take no part */
    std::vector<std::pair<uint64_t, size_t> > hv;
    hv.reserve(np);
    for (size_t p = 0; p < np; p++) {
      if (nodata[p]) { ddup_lookup[p] = NODATA_CLASS; continue; }
      hv.push_back(hp[p]);
    }
    hp.clear();
    std::sort(hv.begin(), hv.end());
    ddup_rep.clear();
    size_t i = 0;
    const size_t vb = nb * sizeof(float);
    while (i < hv.size()) {
      size_t j = i;
      while (j < hv.size() && hv[j].first == hv[i].first) j++;
      /* one hash bucket: assign exact-equality classes within it */
      for (size_t u = i; u < j; u++) {
        size_t pu = hv[u].second;
        if (ddup_lookup[pu] != 0 || true) { /* fall through to search below */ }
        long found = -1;
        for (size_t v = i; v < u; v++) {
          size_t pv = hv[v].second;
          if (std::memcmp(img + pu * nb, img + pv * nb, vb) == 0) {
            found = (long)ddup_lookup[pv];
            break;
          }
        }
        if (found >= 0) ddup_lookup[pu] = (size_t)found;
        else { ddup_lookup[pu] = ddup_rep.size(); ddup_rep.push_back(pu); }
      }
      i = j;
    }
    FILE* f = std::fopen(dedup_cache.c_str(), "wb");
    if (f) {
      cache_write_head(f, sum, (uint64_t)np * 1000003ull + nb);
      size_t nd = ddup_rep.size();
      std::fwrite(&nd, sizeof nd, 1, f);
      std::fwrite(&ddup_rep[0], sizeof(size_t), nd, f);
      std::fwrite(&ddup_lookup[0], sizeof(size_t), np, f);
      std::fclose(f);
    }
  }
  size_t n_ddup = ddup_rep.size();
  std::fprintf(stderr, "[dedup] %zu pixels -> %zu exact equivalence classes\n",
               np - n_nodata, n_ddup);

  /* ---- 4. decimate the deduplicated set by a regular stride -------- */
  size_t nskip = A.nskip > 0 ? (size_t)A.nskip
                             : std::max<size_t>(1, (n_ddup + A.budget_points - 1) / A.budget_points);
  std::vector<size_t> keep; /* indices into ddup_rep */
  for (size_t i = 0; i < n_ddup; i += nskip) keep.push_back(i);
  size_t n = keep.size();
  std::vector<float> pts(n * nb);
  for (size_t i = 0; i < n; i++) {
    const float* src = img + ddup_rep[keep[i]] * nb;
    for (size_t b = 0; b < nb; b++) pts[i * nb + b] = src[b];
  }
  size_t kmax = std::min<size_t>((size_t)std::max(2L, A.kmax), n - 1);
  size_t kstep = (size_t)std::max(1L, A.kstep);
  size_t kmin = kstep;                 /* the sweep starts at the step size */
  if (kmin > kmax) kmin = kmax;
  size_t n_levels = kmax / kstep;      /* one job per level */
  if (n_levels < 1) n_levels = 1;
  size_t min_class = A.min_class > 0 ? (size_t)A.min_class
                                     : (size_t)std::llround(std::sqrt((double)n));
  if (min_class < 2) min_class = 2;
  std::fprintf(stderr,
               "[params] n_skip=%zu -> %zu retained points | K %zu..%zu step %zu "
               "(%zu levels) | min_class=%zu | table %.0f MB\n",
               nskip, n, kmin, kmax, kstep, n_levels, min_class,
               (double)n * kmax * 12.0 / 1048576.0);

  /* ---- 5. neighbour table, memoised ------------------------------- */
  std::vector<float> dd;
  std::vector<size_t> di;
  char tagbuf[128];
  std::snprintf(tagbuf, sizeof tagbuf, "%s.kgc_knn_s%zu_k%zu",
                A.image.c_str(), nskip, kmax);
  str knn_cache(tagbuf);
  uint64_t knn_tag = (uint64_t)nskip * 1000003ull + (uint64_t)kmax;
  loaded = false;
  if (!A.no_cache && file_exists(knn_cache)) {
    FILE* f = std::fopen(knn_cache.c_str(), "rb");
    if (f) {
      size_t nn = 0, kk = 0;
      if (cache_read_head(f, sum, knn_tag) &&
          std::fread(&nn, sizeof nn, 1, f) == 1 &&
          std::fread(&kk, sizeof kk, 1, f) == 1 && nn == n && kk == kmax) {
        dd.resize(n * kmax);
        di.resize(n * kmax);
        if (std::fread(&dd[0], sizeof(float), n * kmax, f) == n * kmax &&
            std::fread(&di[0], sizeof(size_t), n * kmax, f) == n * kmax)
          loaded = true;
      }
      std::fclose(f);
    }
    if (loaded) std::fprintf(stderr, "[neighbours] reusing %s\n", knn_cache.c_str());
  }
  if (!loaded) {
    std::fprintf(stderr, "[neighbours] computing %zu x %zu table (%.0f MB)\n",
                 n, kmax, (double)n * kmax * 12.0 / 1048576.0);
    build_knn(&pts[0], n, nb, kmax, threads, dd, di);
    FILE* f = std::fopen(knn_cache.c_str(), "wb");
    if (f) {
      cache_write_head(f, sum, knn_tag);
      std::fwrite(&n, sizeof n, 1, f);
      std::fwrite(&kmax, sizeof kmax, 1, f);
      std::fwrite(&dd[0], sizeof(float), n * kmax, f);
      std::fwrite(&di[0], sizeof(size_t), n * kmax, f);
      std::fclose(f);
    }
  }

  /* ---- pixels to retained points, once ---------------------------- */
  std::vector<long> leafmap(np, -1);
  std::vector<double> w(n, 0.0), hcount(n, 0.0);
  double HN = 0;
  {
    KdTree t;
    t.build(&pts[0], n, nb, 16);
    parallel_for(np, threads, [&](size_t p, size_t) {
      leafmap[p] = nodata[p] ? -1L : (long)t.nearest(img + p * nb);
    });
    for (size_t p = 0; p < np; p++) {
      long l = leafmap[p];
      if (l < 0) continue;
      w[(size_t)l] += 1.0;
      if (hint[p] > 0.5f) { hcount[(size_t)l] += 1.0; HN += 1.0; }
    }
  }
  std::fprintf(stderr, "[hint] %.0f hint pixels (%.1f%% of the scene)\n",
               HN, 100.0 * HN / (double)np);

  /* ---- 6/7. sweep K in parallel ------------------------------------ */
  g_n = n; g_nb = nb; g_kmax = kmax; g_np = np;
  g_min_class = min_class; g_HN = HN;
  g_dd = &dd[0]; g_di = &di[0]; g_w = &w[0]; g_h = &hcount[0];
  g_scratch.resize(threads);
  for (size_t t = 0; t < threads; t++) {
    g_scratch[t].S.resize(n);
    g_scratch[t].asc.resize(n);
    g_scratch[t].peak.resize(n);
  }
  g_levels.assign(n_levels, Level());
  g_kstep = kstep;
  g_n_levels = n_levels;
  g_patience = (size_t)std::max(0L, A.patience);
  g_best_mi = -1e300;
  g_best_K = -1;
  g_since_best = 0;
  g_n_done = 0;
  g_stop = 0;
  pthread_mutex_init(&g_res_mtx, NULL);

  std::fprintf(stderr, "[sweep] %zu levels of K over %zu worker(s), %s\n",
               n_levels, threads,
               g_patience ? "stopping early once the best stops improving"
                          : "sweeping every level (early stop disabled)");
  parallel_jobs(n_levels, threads, sweep_job, &g_stop);
  pthread_mutex_destroy(&g_res_mtx);
  std::fprintf(stderr, "[sweep] %zu of %zu levels evaluated%s\n",
               g_n_done, n_levels, g_stop ? " (stopped early)" : "");
  long bestK = g_best_K;
  if (bestK < 0) die("no class passed the size and lift guards at any K");
  std::fprintf(stderr, "[select] best K = %ld\n", bestK);
  std::vector<Level>& levels = g_levels;

  /* ---- masks at the chosen level and the two beside it ------------- *
   * The sweep keeps only a handful of scalars per level, so its memory is the
   * same whichever level a worker happens to be on.  Now that the best K is
   * known, the three class maps are simply recomputed from the neighbour
   * table, which is still in memory.                                       */
  std::vector<float> o(np * 6, 0.f);
  std::vector<long> classmap(n, -1);
  size_t Ks[3];
  Ks[0] = (size_t)std::max((long)kstep, bestK - (long)kstep);
  Ks[1] = (size_t)bestK;
  Ks[2] = (size_t)std::min((long)kmax, bestK + (long)kstep);
  int band_of[3] = {4, 0, 5};
  for (int t = 0; t < 3; t++) {
    std::vector<long> cls;
    Level L = compute_level(Ks[t], g_scratch[0], &cls);
    for (size_t p = 0; p < np; p++) {
      long l = leafmap[p];
      o[band_of[t] * np + p] = (l >= 0 && cls[(size_t)l] == L.best) ? 1.f : 0.f;
    }
    if (t == 1) {
      classmap = cls;
      std::fprintf(stderr,
                   "[select] K=%zu: %.0f pixels, %.0f in hint "
                   "(precision %.3f, recall %.3f, MI %.4f, lift %.2f) of %zu classes\n",
                   Ks[t], L.npix, L.nh, L.nh / std::max(1.0, L.npix),
                   L.nh / std::max(1.0, HN), L.mi, L.lift, L.n_classes);
    }
  }

  /* ---- codeword evidence field, a diagnostic ---------------------- */
  double kl = 0;
  {
    std::vector<long> cls;
    compute_level(kmin, g_scratch[0], &cls);
    size_t M = 0;
    for (size_t i = 0; i < n; i++) if (cls[i] + 1 > (long)M) M = (size_t)cls[i] + 1;
    std::vector<double> cH(M, 0.0), cB(M, 0.0);
    for (size_t i = 0; i < n; i++) {
      cH[(size_t)cls[i]] += hcount[i];
      cB[(size_t)cls[i]] += w[i] - hcount[i];
    }
    double sH = 0, sB = 0;
    const double al = 0.5;
    for (size_t m = 0; m < M; m++) { sH += cH[m] + al; sB += cB[m] + al; }
    std::vector<double> llr(M, 0.0);
    for (size_t m = 0; m < M; m++) {
      double pH = (cH[m] + al) / sH, pB = (cB[m] + al) / sB;
      llr[m] = std::log(pH / pB);
      kl += pH * llr[m];
    }
    for (size_t p = 0; p < np; p++) {
      long l = leafmap[p];
      o[2 * np + p] = (l >= 0) ? (float)llr[(size_t)cls[(size_t)l]] : 0.f;
    }
    std::fprintf(stderr,
                 "[hint] codebook = the clustering at K=%zu (%zu classes); "
                 "D_KL(hint||background) = %.4f nats (diagnostic)\n", kmin, M, kl);
  }

  for (size_t p = 0; p < np; p++) {
    long l = leafmap[p];
    o[1 * np + p] = hint[p];
    o[3 * np + p] = (l >= 0) ? (float)classmap[(size_t)l] : -1.f;
  }

  /* ---- outputs ---------------------------------------------------- */
  std::vector<str> bn;
  bn.push_back("selected class at the best K = " + std::to_string(bestK));
  bn.push_back("input hint mask (1 = true / 0 = false)");
  bn.push_back("codeword log-likelihood ratio log(p_hint/p_background) in nats [diagnostic]");
  bn.push_back("class map at K = " + std::to_string(bestK) + " [-1 where excluded]");
  bn.push_back("best class at K = " + std::to_string(Ks[0]));
  bn.push_back("best class at K = " + std::to_string(Ks[2]));
  std::map<str, str> ex;
  ex["kgc best k"] = std::to_string(bestK);
  ex["kgc k below"] = std::to_string(Ks[0]);
  ex["kgc k above"] = std::to_string(Ks[2]);
  ex["kgc k step"] = std::to_string(kstep);
  write_envi_bsq(out + "_selected", &o[0], 6, H, bn, &ex);
  std::fprintf(stderr, "[out] %s_selected.bin\n", out.c_str());

  {
    std::ofstream f((out + "_klevels.csv").c_str());
    f << "K,n_classes,best_class,n_pixels,n_hint,precision,recall,mi,lift\n";
    for (size_t i = 0; i < levels.size(); i++) {
      const Level& L = levels[i];
      if (!L.done) continue;
      f << L.K << "," << L.n_classes << "," << L.best << "," << L.npix << ","
        << L.nh << "," << (L.npix > 0 ? L.nh / L.npix : 0.0) << ","
        << (HN > 0 ? L.nh / HN : 0.0) << "," << L.mi << "," << L.lift << "\n";
    }
  }
  {
    std::ofstream f((out + "_params.txt").c_str());
    f << "image = " << A.image << "\nhint = " << A.hint << "\n";
    f << "samples = " << H.samples << "\nlines = " << H.lines << "\nbands = " << nb << "\n";
    f << "pixels = " << np << "\nnon_finite_pixels = " << n_nodata << "\n";
    f << "precondition = per band (x - min) / (max - min)\n";
    f << "dedup = exact equality of the band vector, whole image, before the stride\n";
    f << "n_dedup = " << n_ddup << "\n";
    f << "n_skip = " << nskip << " # every n_skip-th deduplicated record\n";
    f << "n_points = " << n << "\n";
    f << "k_max = " << kmax << " # neighbours held per point\n";
    f << "k_step = " << kstep << " # the sweep visits K = kstep, 2*kstep, ...\n";
    f << "k_start = " << kmin << "\nmin_class = " << min_class << "\n";
    f << "levels_evaluated = " << g_n_done << " of " << n_levels << "\n";
    f << "stopped_early = " << (g_stop ? "yes" : "no") << "\n";
    f << "patience = " << g_patience << "\n";
    f << "threads = " << threads << "\n";
    f << "n_hint = " << HN << "\nkl_hint_background = " << kl << "\n";
    f << "best_k = " << bestK << "\n";
  }
  std::free(img);
  std::fprintf(stderr, "[done]\n");
  return 0;
}
