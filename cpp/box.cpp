/* 20220526: box filter
Input:
  32-bit IEEE standard floating-point BSQ format stack */
#include"misc.h"
static size_t nrow, ncol, nband, np, m;
static float *out, *dat, t;
static int *bp;
static long int dw;

// thsi should go in misc.h
inline int is_bad(float * dat, size_t i, size_t n_b){
  int zero = true;
  for0(m, n_b){  // find bad/empty pix
    t = dat[np * m + i];
    if(isnan(t) || isinf(t)) return true;
    if(t != 0) zero = false;
  }
  return zero;
}

void filter_line(size_t line_ix){
  if(line_ix % 100 == 0) printf("line %zu\n", line_ix);
  size_t b_ix = line_ix / nrow;  // process a row
  size_t r_ix = line_ix % nrow;
  size_t bk = b_ix * np;
  size_t ki = bk + (r_ix * ncol);
  size_t y, ix, iy;
  float npix, d, dd;
  long int dx, dy, wind;

  for0(y, ncol){
    ix = ki + y; // index of pix at row r_ix and col ix
    out[ix] = npix = d = 0.;

    for(dx = ((long int)r_ix - dw);
        dx <= ((long int)r_ix + dw);
	dx++){

      for(dy = ((long int)y - dw);
          dy <= ((long int)y + dw);
	  dy++){

        iy = dx * ncol + dy;
	if(bp[iy]) continue; // skip bad px
        wind = bk + iy; // for each pixel in window
        if((dx >= 0) && (dy >= 0) && (dx < nrow) && (dy < ncol)){
	  dd = dat[wind];
	  if(!(isnan(dd) || isinf(dd))){
            npix++;
            d += (double) dd;
	  }
        }
      }
    }
    if(npix > 0.)
      out[ix] = (float)(d / ((double)npix));
    else
      out[ix] = dat[ix];
  }
}

int main(int argc, char ** argv){
  if(argc < 3)
  err("box [input binary file name] [window size (odd)] ");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name

  size_t i, j, k, n;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  n = (size_t) atoi(argv[2]);
  if((n - 1) %2 != 0)
  err("odd window size req'd");
  dw = (long int)(n / 2);

  dat = bread(fn, nrow, ncol, nband); // load floats to array
  out = falloc(np * nband); // output data

  bp = ialloc(np);
  for0(i, np)
    bp[i] = is_bad(dat, i, nband);
  
  parfor(0, nband * nrow, filter_line);

  // write output file
  str ofn(fn + str("_box.bin"));
  str ohfn(fn + str("_box.hdr"));

  printf("nr2 %zu nc2 %zu nband %zu\n", nrow, ncol, nband);
  str a(exec((str("cp -v ") + hfn + str(" ") + ohfn)).c_str());
  //hwrite(ohfn, nrow, ncol, nband); // write output header
  bwrite(out, ofn, nrow, ncol, nband);
  return 0;
}
