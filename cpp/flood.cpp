/* 20220225 raster flood fill on mask: background is label 0.
New labels to connected components of image areas valued 1.

20251022: generalized to perform flood fill on arbitrary labels ( int represented in 32-bit float type )

Find connected components represented by contiguous pixels with same value.

NAN is now no-data area. All other pixels are labelled.
*/

#include "misc.h"
#include <cmath>
#include <cstdint>
#include <vector>
#include <stack>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>

using std::size_t;

// Globals (kept similar to original but typed clearly)
size_t *out_labels = nullptr; // output labels (size_t per cell)
float *dat = nullptr; // input data
float *out_f = nullptr; // float output for writing
uint8_t *visited = nullptr; // visited marker (0/1)
size_t i_next = 1;
size_t nrow = 0, ncol = 0, nband = 0;
long int nf = 0;

static inline bool mul_will_overflow_size_t(size_t a, size_t b) {
  if (a == 0 || b == 0) return false;
  return a > (std::numeric_limits<size_t>::max() / b);
}

// Iterative flood-fill starting at (si,sj) for target value label_val
// returns number of pixels flooded
long int flood_iter(long int si, long int sj, float label_val, size_t np) {
  if (si < 0 || sj < 0 || (size_t)si >= nrow || (size_t)sj >= ncol) return 0;
  long int count = 0;
  // stack of linear indices (use vector as stack)
  std::vector<size_t> stack;
  stack.reserve(1024);
  size_t start_idx = (size_t)si * ncol + (size_t)sj;
  // Quick rejects
  if (visited[start_idx]) return 0;

  float vstart = dat[start_idx];
  if (std::isnan(vstart) || vstart != label_val) return 0;

  stack.push_back(start_idx);
  while (!stack.empty()) {
    size_t idx = stack.back();
    stack.pop_back();
    if (visited[idx]) continue; // might have been visited earlier
    float v = dat[idx];
    if (std::isnan(v) || v != label_val) continue;
    visited[idx] = 1;
    out_labels[idx] = i_next;
    count++;
    // compute 2D coords
    size_t ii = idx / ncol;
    size_t jj = idx % ncol;

    // push 8 neighbors (check bounds)
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
          if (!std::isnan(nv) && nv == label_val) {
            stack.push_back(nidx);
          }
        }
      }
    }
  }
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    err("Usage: flood_general.exe [input file name]");
  }

  str fn(argv[1]);
  str ofn(fn + "_cc_labels.bin");
  str hfn(hdr_fn(fn));
  str hofn(hdr_fn(ofn, true));
  str ofn_f(fn + "_cc_labels_float.bin");
  str hofn_f(hdr_fn(ofn_f, true));

  size_t d_type = hread(hfn, nrow, ncol, nband);
  
  if (nband != 1) {
    err("Expected single-band image");
  }
  if (d_type != 4) { // expecting 32-bit float (type 4)
    err("Expected 32-bit float image (type 4)");
  }

  if (nrow == 0 || ncol == 0) {
    err("Image has zero rows or columns");
  }

  // Check multiplication overflow
  if (mul_will_overflow_size_t(nrow, ncol)) {
    err("nrow * ncol would overflow size_t");
  }
  size_t np = nrow * ncol;

  // read data
  dat = bread(fn, nrow, ncol, nband);
  if (!dat) err("bread() failed to read input file");

  // allocate outputs (use calloc-like zero init where appropriate)
  out_labels = (size_t*)alloc(np * sizeof(size_t));
  if (!out_labels) err("alloc() failed for out_labels");
  // zero labels
  memset(out_labels, 0, np * sizeof(size_t));

  visited = (uint8_t*)alloc(np * sizeof(uint8_t));
  if (!visited) err("alloc() failed for visited");
  memset(visited, 0, np * sizeof(uint8_t));

  out_f = falloc(np);
  if (!out_f) err("falloc() failed for out_f");

  i_next = 1;

  for (long int i = 0; i < (long int)nrow; ++i) {
    for (long int j = 0; j < (long int)ncol; ++j) {
      size_t ij = (size_t)i * ncol + (size_t)j;
      if (visited[ij]) continue;
      float val = dat[ij];
      if (std::isnan(val)) {
        visited[ij] = 1; // mark nodata as visited to avoid repeated checks
        out_labels[ij] = NAN;
        continue;
      }
      nf = 0;
      long int flooded = flood_iter(i, j, val, np);
      if (flooded > 0) {
        nf = flooded;
        i_next++;
      }
    }
  }

  // populate float output (use NaN for nodata)
  for (size_t k = 0; k < np; ++k) {
    if (std::isnan(dat[k])) {
      out_f[k] = NAN;
    }
    else{
      // cast label to float, safe if labels fit in float mantissa
      out_f[k] = static_cast<float>(out_labels[k]);
    }
  }

  // Write outputs with checks
  FILE *f_bin = wopen(ofn);
  if (!f_bin) err("wopen() failed: cannot open output binary file");
  if (fwrite(out_labels, sizeof(size_t), np, f_bin) != np) {
    fclose(f_bin);
    err("fwrite failed writing labels");
  }
  fclose(f_bin);

  // NOTE: If you need a specific label size on disk (32-bit vs 64-bit), consider
  // writing uint32_t labels instead of size_t for portability.
  hwrite(hofn, nrow, ncol, 1, (int)(sizeof(size_t) * 8)); // write bit depth = bytes*8

  bwrite(out_f, ofn_f, nrow, ncol, 1);
  hwrite(hofn_f, nrow, ncol, 1, 4); // float output (32-bit)

  // cleanup
  free(dat);
  free(out_labels);
  free(out_f);
  free(visited);

  return 0;
}
