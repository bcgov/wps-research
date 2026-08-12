/* ===========================================================================
 * kgc.h -- Density-Ordered Hill Climbing with Conditional Advancement
 *          Self-contained support library (header-only).
 *
 * Contains:
 *   - ENVI .hdr parsing / writing (type 4 only), BIP / BSQ / BIL input,
 *     BSQ output with map / projection / CRS metadata carried through.
 *   - In-RAM interleave conversion (BSQ->BIP, BIL->BIP).
 *   - A 3..N dimensional k-d tree for k-nearest-neighbour queries.
 *   - Union-find with an explicit binary linkage tree (SciPy-style), so that
 *     every node of the dendrogram is individually addressable.
 *   - Information-theoretic scoring helpers (KL, JS, mutual information).
 *
 * Depends only on the C++11 standard library and <thread>.
 * ===========================================================================*/
#ifndef KGC_H
#define KGC_H

#include <algorithm>
#include <atomic>
#include <functional>
#include <pthread.h>
#include <unistd.h>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

namespace kgc {

typedef std::string str;

/* ------------------------------------------------------------------------ */
/* small utilities                                                           */
/* ------------------------------------------------------------------------ */

inline void die(const str& m) {
  std::fprintf(stderr, "ERROR: %s\n", m.c_str());
  std::exit(1);
}

inline str trim(const str& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == str::npos) return str();
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

inline str lower(str s) {
  for (size_t i = 0; i < s.size(); i++) s[i] = (char)std::tolower((unsigned char)s[i]);
  return s;
}

inline str zero_pad(long v, int w) {
  char b[64];
  std::snprintf(b, sizeof(b), "%0*ld", w, v);
  return str(b);
}

inline bool file_exists(const str& p) {
  std::ifstream f(p.c_str());
  return f.good();
}

/* Strip a trailing .bin/.hdr/.dat and return the stem. */
inline str strip_ext(const str& p) {
  size_t d = p.find_last_of('.');
  size_t s = p.find_last_of("/\\");
  if (d != str::npos && (s == str::npos || d > s)) return p.substr(0, d);
  return p;
}

/* Given a data file path, guess the matching header path. */
inline str hdr_path_for(const str& data_path) {
  str a = strip_ext(data_path) + ".hdr";
  if (file_exists(a)) return a;
  str b = data_path + ".hdr";
  if (file_exists(b)) return b;
  return a;
}

/* ------------------------------------------------------------------------ */
/* ENVI header                                                               */
/* ------------------------------------------------------------------------ */

struct EnviHeader {
  size_t samples, lines, bands;
  int data_type;          /* 4 == 32-bit IEEE float, the only type supported  */
  int byte_order;         /* 0 == little endian                              */
  size_t header_offset;
  str interleave;         /* "bip" | "bsq" | "bil"                           */
  str description;
  str file_type;
  std::vector<str> band_names;
  /* Georeferencing / projection fields carried through verbatim.            */
  std::map<str, str> geo;     /* key -> raw value (without braces stripped)  */
  std::map<str, str> extra;   /* any other field we did not interpret        */

  EnviHeader()
      : samples(0), lines(0), bands(0), data_type(4), byte_order(0),
        header_offset(0), interleave("bsq"), file_type("ENVI Standard") {}

  size_t npix() const { return samples * lines; }
};

/* Fields we copy verbatim from input to every raster we emit. Anything a
 * reader needs to place the product on the map lives here. Absent fields are
 * simply skipped -- a file with no georeferencing is not an error.          */
inline const char** geo_field_names(int* n) {
  static const char* f[] = {
      "map info", "projection info", "coordinate system string",
      "pixel size", "datum", "x start", "y start", "sensor type",
      "wavelength units", "reflectance scale factor", "geo points",
      "rpc info", "spatialref"};
  *n = (int)(sizeof(f) / sizeof(f[0]));
  return f;
}

inline bool is_geo_field(const str& k) {
  int n;
  const char** f = geo_field_names(&n);
  for (int i = 0; i < n; i++)
    if (k == f[i]) return true;
  return false;
}

/* Split "a, b, c" honouring nothing but commas (band names never contain
 * commas in practice; map info values are kept whole so we never split them).*/
inline std::vector<str> split_commas(const str& s) {
  std::vector<str> out;
  str cur;
  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] == ',') {
      out.push_back(trim(cur));
      cur.clear();
    } else
      cur += s[i];
  }
  out.push_back(trim(cur));
  return out;
}

inline EnviHeader read_envi_header(const str& path) {
  std::ifstream f(path.c_str());
  if (!f.good()) die("cannot open ENVI header: " + path);
  EnviHeader h;
  str line, all;
  {
    std::stringstream ss;
    ss << f.rdbuf();
    all = ss.str();
  }
  /* Tokenise into "key = value" records, where a value may span lines when
   * wrapped in { }.                                                        */
  size_t i = 0;
  /* Skip the leading "ENVI" magic. */
  while (i < all.size() && all[i] != '\n') i++;
  while (i < all.size()) {
    /* read a key */
    size_t eq = all.find('=', i);
    if (eq == str::npos) break;
    str key = lower(trim(all.substr(i, eq - i)));
    size_t vs = eq + 1;
    while (vs < all.size() && (all[vs] == ' ' || all[vs] == '\t')) vs++;
    str value;
    if (vs < all.size() && all[vs] == '{') {
      size_t close = all.find('}', vs);
      if (close == str::npos) close = all.size() - 1;
      value = all.substr(vs + 1, close - vs - 1);
      i = close + 1;
      while (i < all.size() && all[i] != '\n') i++;
      if (i < all.size()) i++;
    } else {
      size_t nl = all.find('\n', vs);
      if (nl == str::npos) nl = all.size();
      value = all.substr(vs, nl - vs);
      i = nl < all.size() ? nl + 1 : all.size();
    }
    str v = trim(value);
    if (key == "samples")
      h.samples = (size_t)atol(v.c_str());
    else if (key == "lines")
      h.lines = (size_t)atol(v.c_str());
    else if (key == "bands")
      h.bands = (size_t)atol(v.c_str());
    else if (key == "data type")
      h.data_type = atoi(v.c_str());
    else if (key == "byte order")
      h.byte_order = atoi(v.c_str());
    else if (key == "header offset")
      h.header_offset = (size_t)atol(v.c_str());
    else if (key == "interleave")
      h.interleave = lower(v);
    else if (key == "description")
      h.description = v;
    else if (key == "file type")
      h.file_type = v;
    else if (key == "band names") {
      h.band_names = split_commas(v);
    } else if (is_geo_field(key))
      h.geo[key] = v;
    else
      h.extra[key] = v;
  }
  if (h.samples == 0 || h.lines == 0 || h.bands == 0)
    die("ENVI header missing samples/lines/bands: " + path);
  return h;
}

/* Write a BSQ, data-type-4 header cloning the georeferencing of `src`.      */
inline void write_envi_header(const str& path, const EnviHeader& src,
                              size_t bands, const std::vector<str>& band_names,
                              const std::map<str, str>* extra_fields = 0) {
  std::ofstream f(path.c_str());
  if (!f.good()) die("cannot write ENVI header: " + path);
  f << "ENVI\n";
  f << "description = {\n" << path << "}\n";
  f << "samples = " << src.samples << "\n";
  f << "lines   = " << src.lines << "\n";
  f << "bands   = " << bands << "\n";
  f << "header offset = 0\n";
  f << "file type = ENVI Standard\n";
  f << "data type = 4\n";
  f << "interleave = bsq\n";
  f << "byte order = 0\n";
  /* Carry georeferencing through verbatim when the source had it. */
  int nf;
  const char** gf = geo_field_names(&nf);
  for (int i = 0; i < nf; i++) {
    std::map<str, str>::const_iterator it = src.geo.find(str(gf[i]));
    if (it != src.geo.end() && !it->second.empty())
      f << gf[i] << " = {" << it->second << "}\n";
  }
  if (extra_fields)
    for (std::map<str, str>::const_iterator it = extra_fields->begin();
         it != extra_fields->end(); ++it)
      f << it->first << " = " << it->second << "\n";
  f << "band names = {\n";
  for (size_t i = 0; i < bands; i++) {
    str n = i < band_names.size() ? band_names[i] : (str("band ") + zero_pad((long)i + 1, 3));
    f << n;
    if (i + 1 < bands) f << ",\n";
  }
  f << "}\n";
}

inline void byteswap32(float* p, size_t n) {
  uint32_t* u = (uint32_t*)p;
  for (size_t i = 0; i < n; i++) {
    uint32_t v = u[i];
    u[i] = ((v >> 24) & 0xffu) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) |
           ((v << 24) & 0xff000000u);
  }
}

/* ------------------------------------------------------------------------ */
/* interleave conversion                                                     */
/* ------------------------------------------------------------------------ */

/* BSQ [band][pixel] -> BIP [pixel][band].  Blocked to stay cache-friendly:
 * a naive transpose of a 300k x N array thrashes on every store.           */
inline void bsq_to_bip(const float* src, float* dst, size_t npix, size_t nb) {
  const size_t BL = 4096;
  for (size_t p0 = 0; p0 < npix; p0 += BL) {
    size_t p1 = std::min(p0 + BL, npix);
    for (size_t b = 0; b < nb; b++) {
      const float* s = src + b * npix + p0;
      float* d = dst + p0 * nb + b;
      for (size_t p = p0; p < p1; p++, s++, d += nb) *d = *s;
    }
  }
}

/* BIL [line][band][sample] -> BIP */
inline void bil_to_bip(const float* src, float* dst, size_t ns, size_t nl,
                       size_t nb) {
  for (size_t l = 0; l < nl; l++)
    for (size_t b = 0; b < nb; b++) {
      const float* s = src + (l * nb + b) * ns;
      float* d = dst + (l * ns) * nb + b;
      for (size_t x = 0; x < ns; x++, s++, d += nb) *d = *s;
    }
}

/* BIP -> BSQ (used on the way out; every product we emit is BSQ).          */
inline void bip_to_bsq(const float* src, float* dst, size_t npix, size_t nb) {
  const size_t BL = 4096;
  for (size_t p0 = 0; p0 < npix; p0 += BL) {
    size_t p1 = std::min(p0 + BL, npix);
    for (size_t b = 0; b < nb; b++) {
      const float* s = src + p0 * nb + b;
      float* d = dst + b * npix + p0;
      for (size_t p = p0; p < p1; p++, s += nb, d++) *d = *s;
    }
  }
}

/* Read an ENVI image and return it in BIP order regardless of how it was
 * stored on disk.  BIP is the layout the clustering wants: the distance
 * kernel touches all nb components of one pixel consecutively.             */
inline float* read_envi_bip(const str& data_path, EnviHeader& h) {
  h = read_envi_header(hdr_path_for(data_path));
  if (h.data_type != 4)
    die("only ENVI data type 4 (32-bit float) is supported; got type " +
        std::to_string(h.data_type));
  size_t n = h.npix() * h.bands;
  std::ifstream f(data_path.c_str(), std::ios::binary);
  if (!f.good()) die("cannot open image: " + data_path);
  f.seekg((std::streamoff)h.header_offset);
  float* raw = (float*)std::malloc(n * sizeof(float));
  if (!raw) die("out of memory reading " + data_path);
  f.read((char*)raw, (std::streamsize)(n * sizeof(float)));
  if ((size_t)f.gcount() != n * sizeof(float))
    die("short read on " + data_path + " (truncated or wrong dimensions)");
  if (h.byte_order == 1) byteswap32(raw, n);
  if (h.interleave == "bip" || h.bands == 1) return raw;
  float* bip = (float*)std::malloc(n * sizeof(float));
  if (!bip) die("out of memory converting interleave");
  if (h.interleave == "bsq")
    bsq_to_bip(raw, bip, h.npix(), h.bands);
  else if (h.interleave == "bil")
    bil_to_bip(raw, bip, h.samples, h.lines, h.bands);
  else
    die("unknown interleave: " + h.interleave);
  std::free(raw);
  return bip;
}

/* Write a BSQ product: `data` is already band-sequential, nb bands.        */
inline void write_envi_bsq(const str& out_base, const float* bsq, size_t nb,
                           const EnviHeader& src,
                           const std::vector<str>& band_names,
                           const std::map<str, str>* extra_fields = 0) {
  str bin = out_base + ".bin";
  std::ofstream f(bin.c_str(), std::ios::binary);
  if (!f.good()) die("cannot write " + bin);
  f.write((const char*)bsq, (std::streamsize)(src.npix() * nb * sizeof(float)));
  f.close();
  write_envi_header(out_base + ".hdr", src, nb, band_names, extra_fields);
}

/* ------------------------------------------------------------------------ */
/* k-d tree (static, median split)                                           */
/* ------------------------------------------------------------------------ */

/* Exact k-nearest-neighbour queries in d dimensions.  For the band counts
 * that occur in raster work (3..30) this turns the O(n^2) neighbour graph
 * into something near O(n k log n), which is what makes full-resolution
 * runs practical.                                                          */
class KdTree {
 public:
  KdTree() : dim_(0), pts_(0), n_(0), leaf_(16) {}

  void build(const float* pts, size_t n, size_t dim, size_t leaf = 16) {
    pts_ = pts;
    n_ = n;
    dim_ = dim;
    leaf_ = leaf;
    idx_.resize(n);
    for (size_t i = 0; i < n; i++) idx_[i] = i;
    nodes_.clear();
    nodes_.reserve(2 * (n / leaf_ + 1) + 1);
    build_rec(0, n);
  }

  /* Fill out_d[0..k-1], out_i[0..k-1] with the k nearest neighbours of the
   * query point `q`, sorted ascending, excluding `self` if >= 0.           */
  void knn(const float* q, size_t k, long self, float* out_d,
           size_t* out_i) const {
    std::priority_queue<std::pair<float, size_t> > heap; /* max-heap on dist */
    search(0, q, k, self, heap);
    size_t m = heap.size();
    for (size_t r = m; r > 0; r--) {
      out_d[r - 1] = std::sqrt(heap.top().first);
      out_i[r - 1] = heap.top().second;
      heap.pop();
    }
    for (size_t r = m; r < k; r++) { /* degenerate: fewer than k points */
      out_d[r] = out_d[m ? m - 1 : 0];
      out_i[r] = out_i[m ? m - 1 : 0];
    }
  }

  size_t nearest(const float* q) const {
    std::priority_queue<std::pair<float, size_t> > heap;
    search(0, q, 1, -1, heap);
    return heap.empty() ? 0 : heap.top().second;
  }

 private:
  struct Node {
    float split;
    size_t left, right; /* child node ids, 0 == none            */
    size_t begin, end;  /* index range for leaves               */
    int axis;           /* -1 for a leaf                        */
  };

  size_t build_rec(size_t begin, size_t end) {
    size_t id = nodes_.size();
    nodes_.push_back(Node());
    Node nd;
    nd.begin = begin;
    nd.end = end;
    nd.left = nd.right = 0;
    nd.axis = -1;
    nd.split = 0.f;
    if (end - begin > leaf_) {
      /* split on the widest axis at the median */
      int best = 0;
      float bestw = -1.f;
      for (size_t a = 0; a < dim_; a++) {
        float lo = std::numeric_limits<float>::max(), hi = -lo;
        for (size_t i = begin; i < end; i++) {
          float v = pts_[idx_[i] * dim_ + a];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        if (hi - lo > bestw) {
          bestw = hi - lo;
          best = (int)a;
        }
      }
      if (bestw > 0.f) {
        size_t mid = begin + (end - begin) / 2;
        const float* p = pts_;
        size_t d = dim_;
        int a = best;
        std::nth_element(idx_.begin() + begin, idx_.begin() + mid,
                         idx_.begin() + end,
                         [p, d, a](size_t x, size_t y) {
                           return p[x * d + a] < p[y * d + a];
                         });
        nd.axis = best;
        nd.split = pts_[idx_[mid] * dim_ + best];
        nodes_[id] = nd;
        size_t l = build_rec(begin, mid);
        size_t r = build_rec(mid, end);
        nodes_[id].left = l;
        nodes_[id].right = r;
        return id;
      }
    }
    nodes_[id] = nd;
    return id;
  }

  void search(size_t id, const float* q, size_t k, long self,
              std::priority_queue<std::pair<float, size_t> >& heap) const {
    const Node& nd = nodes_[id];
    if (nd.axis < 0) {
      for (size_t i = nd.begin; i < nd.end; i++) {
        size_t j = idx_[i];
        if ((long)j == self) continue;
        const float* p = pts_ + j * dim_;
        float s = 0.f;
        for (size_t a = 0; a < dim_; a++) {
          float df = p[a] - q[a];
          s += df * df;
        }
        if (heap.size() < k)
          heap.push(std::make_pair(s, j));
        else if (s < heap.top().first) {
          heap.pop();
          heap.push(std::make_pair(s, j));
        }
      }
      return;
    }
    float diff = q[nd.axis] - nd.split;
    size_t near = diff < 0.f ? nd.left : nd.right;
    size_t far = diff < 0.f ? nd.right : nd.left;
    search(near, q, k, self, heap);
    if (heap.size() < k || diff * diff < heap.top().first)
      search(far, q, k, self, heap);
  }

  size_t dim_;
  const float* pts_;
  size_t n_;
  size_t leaf_;
  std::vector<size_t> idx_;
  std::vector<Node> nodes_;
};

/* ------------------------------------------------------------------------ */
/* content checksum, for cache validity                                      */
/* ------------------------------------------------------------------------ */

/* FNV-1a over the raw bytes of a file.  Used to decide whether a memoised
 * deduplication or neighbour table still matches the image it was built from;
 * only the binary is hashed, never the header.                             */
inline uint64_t file_checksum(const str& path) {
  std::ifstream f(path.c_str(), std::ios::binary);
  if (!f.good()) return 0;
  uint64_t h = 1469598103934665603ull;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(&buf[0], (std::streamsize)buf.size());
    std::streamsize got = f.gcount();
    for (std::streamsize i = 0; i < got; i++) {
      h ^= (uint64_t)(unsigned char)buf[(size_t)i];
      h *= 1099511628211ull;
    }
  }
  return h;
}

/* ------------------------------------------------------------------------ */
/* information theory                                                        */
/* ------------------------------------------------------------------------ */

inline double xlogx_ratio(double p, double q) {
  if (p <= 0.0) return 0.0;
  if (q <= 0.0) return 0.0;
  return p * std::log(p / q);
}

/* Kullback-Leibler divergence D(p||q) in nats, distributions given as
 * unnormalised counts with additive (Laplace/Krichevsky-Trofimov) smoothing.*/
inline double kl_divergence(const std::vector<double>& p_cnt,
                            const std::vector<double>& q_cnt, double alpha) {
  size_t m = p_cnt.size();
  double ps = 0, qs = 0;
  for (size_t i = 0; i < m; i++) {
    ps += p_cnt[i] + alpha;
    qs += q_cnt[i] + alpha;
  }
  double d = 0;
  for (size_t i = 0; i < m; i++) {
    double p = (p_cnt[i] + alpha) / ps, q = (q_cnt[i] + alpha) / qs;
    d += xlogx_ratio(p, q);
  }
  return d;
}

/* Jensen-Shannon divergence: symmetric, bounded by log 2, finite even when
 * the supports differ -- which they routinely do here.                     */
inline double js_divergence(const std::vector<double>& p_cnt,
                            const std::vector<double>& q_cnt, double alpha) {
  size_t m = p_cnt.size();
  double ps = 0, qs = 0;
  for (size_t i = 0; i < m; i++) {
    ps += p_cnt[i] + alpha;
    qs += q_cnt[i] + alpha;
  }
  double d = 0;
  for (size_t i = 0; i < m; i++) {
    double p = (p_cnt[i] + alpha) / ps, q = (q_cnt[i] + alpha) / qs;
    double r = 0.5 * (p + q);
    d += 0.5 * xlogx_ratio(p, r) + 0.5 * xlogx_ratio(q, r);
  }
  return d;
}

inline double binary_entropy(double p) {
  if (p <= 0.0 || p >= 1.0) return 0.0;
  return -(p * std::log(p) + (1 - p) * std::log(1 - p));
}

/* Mutual information of the 2x2 table
 *        in H     not H
 *   in C   a        b
 *  not C   c        d
 * in nats. This is the expected KL divergence between the joint and the
 * product of marginals, and is the quantity we maximise when matching a
 * candidate cluster to the hint.                                           */
inline double mutual_information_2x2(double a, double b, double c, double d) {
  double n = a + b + c + d;
  if (n <= 0) return 0.0;
  double r1 = a + b, r2 = c + d, c1 = a + c, c2 = b + d;
  double mi = 0;
  const double t[4] = {a, b, c, d};
  const double rr[4] = {r1, r1, r2, r2};
  const double cc[4] = {c1, c2, c1, c2};
  for (int i = 0; i < 4; i++) {
    if (t[i] <= 0) continue;
    double pij = t[i] / n, pi = rr[i] / n, pj = cc[i] / n;
    if (pi <= 0 || pj <= 0) continue;
    mi += pij * std::log(pij / (pi * pj));
  }
  return mi;
}

/* ------------------------------------------------------------------------ */
/* worker pool                                                               */
/* ------------------------------------------------------------------------ */

/* One POSIX thread per CPU.  Work is handed out as integer job indices from a
 * single counter guarded by a mutex: a worker locks, takes the next index,
 * unlocks, and runs that job, repeating until the indices are exhausted or a
 * stop flag is raised.  Jobs are therefore claimed in order but may complete
 * out of order, and no worker ever waits for another.
 *
 * A job is whatever unit of work is natural for the step: one row of the
 * neighbour table, one block of pixels, or one value of K.                  */

inline size_t cpu_count() {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (size_t)n : 1;
}

typedef std::function<void(size_t /*job*/, size_t /*worker*/)> JobFn;

struct PoolCtx {
  pthread_mutex_t* mtx;
  size_t* next;
  size_t n_jobs;
  volatile int* stop;
  const JobFn* body;
  size_t worker;
};

inline void* pool_worker(void* arg) {
  PoolCtx* c = (PoolCtx*)arg;
  for (;;) {
    if (c->stop && *c->stop) return NULL;
    pthread_mutex_lock(c->mtx);
    size_t j = (*c->next)++;
    pthread_mutex_unlock(c->mtx);
    if (j >= c->n_jobs) return NULL;
    (*c->body)(j, c->worker);
  }
}

/* Run `n_jobs` jobs across `nthreads` workers.  If `stop` is non-null the
 * workers check it before claiming each job, so raising it from inside a job
 * drains the pool without cancelling work already in flight.                */
inline void parallel_jobs(size_t n_jobs, size_t nthreads, JobFn body,
                          volatile int* stop = 0) {
  if (n_jobs == 0) return;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > n_jobs) nthreads = n_jobs;
  if (nthreads == 1) {
    for (size_t j = 0; j < n_jobs; j++) {
      if (stop && *stop) break;
      body(j, 0);
    }
    return;
  }
  pthread_mutex_t mtx;
  pthread_mutex_init(&mtx, NULL);
  size_t next = 0;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  std::vector<PoolCtx> ctx(nthreads);
  std::vector<pthread_t> th(nthreads);
  for (size_t t = 0; t < nthreads; t++) {
    ctx[t].mtx = &mtx;
    ctx[t].next = &next;
    ctx[t].n_jobs = n_jobs;
    ctx[t].stop = stop;
    ctx[t].body = &body;
    ctx[t].worker = t;
    pthread_create(&th[t], &attr, pool_worker, (void*)&ctx[t]);
  }
  for (size_t t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&mtx);
}

/* Convenience wrapper for loops over a flat range, one block of `chunk`
 * iterations per job, so that very short bodies do not pay for a lock each.  */
template <typename F>
inline void parallel_for(size_t n, size_t nthreads, F body, size_t chunk = 64) {
  if (n == 0) return;
  size_t n_blocks = (n + chunk - 1) / chunk;
  parallel_jobs(n_blocks, nthreads, [&](size_t b, size_t w) {
    size_t lo = b * chunk, hi = std::min(lo + chunk, n);
    for (size_t i = lo; i < hi; i++) body(i, w);
  });
}

}  /* namespace kgc */

#endif /* KGC_H */
