/* 20210527 raster_mode_filter.cpp from a raster, in each
dimension bin the values in a window. This algorithm then
smooths by interpolating, not by choosing. use

class_mode_filter.cpp for smoothing a class map. Use this
for smoothing a continuous raster

In this implementation a histogram is calculated in each
dimension separately. In another version we could calculate
the histogram on a n-dimensional rectangular grid. In
yet another dimension we could use KGC 2010 (knn graph 
clustering) algorithm in the window. 

20220517 bugfix */
#include"misc.h"

int main(int argc, char ** argv){
  size_t ws;
  float n_bin;
  long int nbin, dw;
  (str fn(""), str hfn(""));

  if(argc < 4) {
    err("mode_filter [input binary file] [window size] [number of bins]");
    ws = 3; //atoi(argv[2]);
    nbin = 2; //atoi(argv[3]);
    n_bin = (float)nbin; // bin size
    if(ws % 2 != 1) err("window size must be odd number");
    dw = (ws - 1) / 2;
    fn = str("test.bin");
    hfn = str(hdr_fn(fn)); // auto-detect header file name
  }
  else{
    ws = atoi(argv[2]);
    nbin = atoi(argv[3]);
    n_bin = (float)nbin; // bin size
    if(ws % 2 != 1) err("window size must be odd number");
    dw = (ws - 1) / 2;
    fn = str(argv[1]); // input file name
    hfn = str(hdr_fn(fn)); // auto-detect header file name
  }
  str out_fn(fn + str("_rmf.bin"));
  str out_hf(fn + str("_rmf.hdr"));

  size_t nr, nc, nb;
  long int nrow, ncol, nband, np, k, n, ci;
  hread(hfn, nr, nc, nb);
  (nrow = nr, ncol = nc, nband = nb);
  np = nrow * ncol; // number of input pix

  float d, * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(np * nband * sizeof(float));
  long int ix, iy, di, dj, dx, dy, dk, i, j;

  float * mn = falloc(nband); // window min this band
  float * mx = falloc(nband); // window max this band
  float * w = falloc(nband);
  float * c = falloc(nbin * nband); // count in "bin m" of for kth dimension at posn: k * nbin + m

  for0(i, nrow){
    if(i % 37 == 0) printf("row %d of %d\n", i, nrow);
    for0(j, ncol){

      for0(k, nband)
        (mn[k] = FLT_MAX, mx[k] = -FLT_MAX); // yes!

      // calculate min, max in window
      for(di = -dw; di <= dw; di++){

        dx = i + di;
        if(dx > nrow || dx < 0) continue;
        
	ix = dx * ncol;
        for(dj = -dw; dj <= dw; dj++){
          
	  dy = j + dj;
          if(dy > ncol || dy < 0) continue;

          for0(k, nband){

            dk = np * k;
            d = dat[ix + dy + dk];
            if(!(isinf(d) || isnan(d))){
              if(d < mn[k]) mn[k] = d;
              if(d > mx[k]) mx[k] = d;
            }
          }
        }
      }

      for0(k, nband)
        (c[k] = 0, w[k] = mx[k] - mn[k]); // w = metawindow width

      // put stuff into bins
      for(di = -dw; di <= dw; di++){
        
	dx = i + di;
        if(dx > nrow || dx < 0) continue;
        
	ix = dx * ncol;
        for(dj = -dw; dj <= dw; dj++){
          
	  dy = j + dj;
          if(dy > ncol || dy < 0) continue;

          for0(k, nband){
            if(w[k] == 0.) continue;

            dk = np * k;
            d = dat[ix + dy + dk];

            if(!(isinf(d) || isnan(d))){
	      ci = (long int) (double)floor( (double)n_bin * (d - mn[k]) / w[k]);
              
	      if(ci < 0 || ci > nbin)
	        err("invalid ci");
              else
                c[k * nbin + ci] += 1;
            }
          }
        }
      }
      // assign a value based on the greatest count. Don't test uniqueness
      for0(k, nband){
        long int xi = (np * k) + (i * ncol) + j;
        if(w[k] > 0){
          long int max_i = 0;
          float max_c = FLT_MIN;
          for0(di, nbin){
            if(c[k * nbin + di] > max_c){
              max_c = c[k * nbin + di];
              max_i = di;
            }
          }
          out[xi] = (((float)max_i) + .5) * w[k] + mn[k];
        }
        else
          out[xi] = mn[k]; // box has same values in it.
      }
    }
  }
  for0(k, np*nband) out[k];
  hwrite(out_hf, nrow, ncol, nband); // write output header
  str os("out.bin");
  bwrite(out, out_fn, nr, nc, nb);

  //free(dat);
  //free(out);
  return 0;
}
