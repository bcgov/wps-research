/* 20220225 raster flood fill on mask: background is label 0.
  New labels to connected components of image areas valued 1.

  20251022: generalized to perform flood fill on arbitrary labels ( int represented in 32-bit float type ) 

Find connected components represented by contiguous pixels with same value.

NAN is now no-data area. All other pixels are labelled.
*/
#include"misc.h"
#include <cmath>  // for isnan()

size_t *out;        // output labels
float *dat, *out_f; // input data, float output for writing
int *visited;       // visited marker
size_t i_next;      // next label
size_t nrow, ncol, nband;
long int nf;        // number of flooded pixels

long int flood(long int i, long int j, float label_val) {
    if (i < 0 || j < 0 || i >= nrow || j >= ncol)
        return 0;

    long int ij = i * ncol + j;
    float d = dat[ij];
    if (visited[ij] || isnan(d) || d != label_val)
        return 0;

    visited[ij] = true;
    out[ij] = i_next;
    nf++;

    long int ret = 1;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0)
                continue;
            int ii = i + di;
            int jj = j + dj;
            if (ii >= 0 && jj >= 0 && ii < nrow && jj < ncol) {
                ret += flood(ii, jj, label_val);
            }
        }
    }
    return ret;
}

int main(int argc, char **argv) {
    if (argc < 2)
        err("Usage: flood_general.exe [input file name]");

    str fn(argv[1]);
    str ofn(fn + "_cc_labels.bin");
    str hfn(hdr_fn(fn));
    str hofn(hdr_fn(ofn, true));
    str ofn_f(fn + "_cc_labels_float.bin");
    str hofn_f(hdr_fn(ofn_f), true);

    size_t d_type = hread(hfn, nrow, ncol, nband);
    if (nband != 1)
        err("Expected single-band image");
    if (d_type != 4)
        err("Expected 32-bit float image (type 4)");

    size_t np = nrow * ncol;
    dat = bread(fn, nrow, ncol, nband);
    out = (size_t *)alloc(np * sizeof(size_t));
    visited = (int *)alloc(np * sizeof(int));
    out_f = falloc(np);

    for (size_t i = 0; i < np; i++) {
        visited[i] = false;
        out[i] = 0;
    }

    i_next = 1;

    for (long int i = 0; i < (long int)nrow; i++) {
        for (long int j = 0; j < (long int)ncol; j++) {
            long int ij = i * ncol + j;
            float val = dat[ij];
            if (!visited[ij] && !isnan(val)) {
                nf = 0;
                flood(i, j, val);
                if (nf > 0)
                    i_next++;
            }
        }
    }

    // Fill float output
    for (size_t i = 0; i < np; i++) {
        if (isnan(dat[i])) {
            out_f[i] = NAN;
        } else {
            out_f[i] = (float)out[i];
        }
    }

    // Write outputs
    FILE *f_bin = wopen(ofn);
    fwrite(out, sizeof(size_t), np, f_bin);
    fclose(f_bin);
    hwrite(hofn, nrow, ncol, 1, 16); // size_t output

    bwrite(out_f, ofn_f, nrow, ncol, 1);
    hwrite(hofn_f, nrow, ncol, 1, 4); // float output

    // Cleanup
    free(dat);
    free(out);
    free(out_f);
    free(visited);
    return 0;
}


