/* 20220526: box filter 
Input: 
32-bit IEEE standard floating-point BSQ format stack */

#include"misc.h"

static float *out, *dat;
static size_t nrow, ncol, nband, np;
static int dw;

void filter_line(size_t line_ix){
  size_t b_ix = line_ix / nrow;  // process a row
  size_t r_ix = line_ix % nrow;
  //printf("ix %zu band %zu row %zu\n", line_ix, b_ix, r_ix);
  size_t bk = b_ix * np;
  size_t ki = bk + (r_ix * ncol);
  size_t y, ix;
  float npix, d;
  long int dx, dy, wind;

  for0(y, ncol){
    ix = ki + y;
    out[ix] = npix = d = 0.;
    for(dx = (r_ix - dw); dx <= (r_ix + dw); dx++){
      for(dy = (y - dw); dy <= (y + dw); dy++){
        wind = bk + (dx * ncol) + dy; // for each pixel in window
        if((dx >= 0) && (dy >= 0) && (dx < nrow) && (dy < ncol)){
          npix++;
          d += (double) dat[wind];
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
  dw = n / 2;

  dat = bread(fn, nrow, ncol, nband); // load floats to array
  out = falloc(np * nband); // output data

  parfor(0, nband * nrow, filter_line);

  // write output file
  str ofn(fn + str("_box.bin"));
  str ohfn(fn + str("_box.hdr"));

  printf("nr2 %zu nc2 %zu nband %zu\n", nrow, ncol, nband);
  hwrite(ohfn, nrow, ncol, nband); // write output header
  bwrite(out, ofn, nrow, ncol, nband);
  return 0;
}
