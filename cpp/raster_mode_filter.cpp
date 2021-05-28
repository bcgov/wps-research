/* 20210527 raster_mode_filter.cpp from a raster, in each
dimension bin the values in a window. This algorithm then
smooths by interpolating, not by choosing. use

class_mode_filter.cpp for smoothing a class map. Use this
for smoothing a continuous raster

In this implementation a histogram is calculated in each
dimension separately. In another version we could calculate
the histogram on a n-dimensional rectangular grid. In 
yet another dimension we could use KGC algorithm in our
window. */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4) err("mode_filter [input binary file] [window size] [number of bins]");
  size_t ws = atoi(argv[2]);

  if(ws % 2 != 1) err("window size must be odd number");
  int dw = (ws - 1) / 2;

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  
  size_t nrow, ncol, nband, np, k, n;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  float d, * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(np * nband * sizeof(float));
  long int ix, iy, di, dj, dx, dy, i, j;

  float mn; // = falloc(nband); // window min this band
  float mx; // = falloc(nband); // window max this band
  float * c = falloc(nbin); // count in "bin m" of for kth dimension at posn: k * nbin + m

  for0(i, nrow){
    for0(j, ncol){
      for0(k, nband){

        for(di = -dw; di <= dw; di++){
          dx = i + di;
          if(dx > nrow || dx < 0) continue;
          ix = dx * ncol;

          for(dj = -dw; dj <= dw; dj++){
            dy = j + dj;
            if(dy > ncol || dy < 0) continue;

            // --
            // calculate the min, max on every window (in every dimension). the min, max on the window for that dimension gets divided up into n_bins bins. Then we do a count on those bins.
            // if there is not a unique max, leave the value unchanged?

            d = dat[ix + dy];
            if(!(isinf(d) || isnan(d))){
            }
 
            // --
          }
        }
      }
    }
  }

  str ofn(fn + str("_mode_filter.bin")); // write output file
  str ohfn(fn + str("_mode_filter.hdr"));
  hwrite(ohfn, nrow, ncol, nband); // write output header
  cout << "+w " << ofn << endl;

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(out, sizeof(float) * nrow * ncol, 1, f); // write data
  fclose(f);

  free(dat);
  free(out);
  return 0;
}
