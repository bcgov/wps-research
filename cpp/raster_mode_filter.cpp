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

20220517 bugfix. Could modify to detect all-zero pxls
too */
#include"misc.h"

int main(int argc, char ** argv){
  size_t ws, nr, nc, nb;
  float n_bin, max_c;
  long int xi, max_i, nbin, dw, nrow, ncol, nband, np, k, n, ci, ix, iy, di, dj, dx, dy, dk, i, j;
  str fn("");
  str hfn("");

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

  hread(hfn, nr, nc, nb);
  (nrow = nr, ncol = nc, nband = nb);
  np = nrow * ncol; // number of input pix

  float d, * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(np * nband * sizeof(float));
  float * mn = falloc(nband); // window min this band
  float * mx = falloc(nband); // window max this band
  float * w = falloc(nband);
  float * c = falloc(nbin * nband); // count in "bin m" of for kth dimension at posn: k * nbin + m

  for0(i, nrow){
    if(i % 37 == 0) printf("row %d of %d\n", i, nrow);
    for0(j, ncol){

      // calculate max and min, window around pxl (i, j) on wndw elements that are in-bounds
      for0(k, nband)
        (mn[k] = FLT_MAX, mx[k] = -FLT_MAX); // yes!

      for(di = -dw; di <= dw; di++){
        dx = i + di;
        if(dx >= nrow || dx < 0) continue;  // skip out of bounds row
        
	ix = dx * ncol;  // row contribution to data idx for pixel in windw

        for(dj = -dw; dj <= dw; dj++){
          
	  dy = j + dj;  // col contribution to data ix for pixel in windw
          if(dy >= ncol || dy < 0) continue;  // skip out of bounds col i.e. pixel (now I get pix vs line!)

          for0(k, nband){
            dk = np * k;  // band contribution to idx of pixel in wndw
            d = dat[ix + dy + dk];
            if(!(isinf(d) || isnan(d))){
              if(d < mn[k]) mn[k] = d;
              if(d > mx[k]) mx[k] = d;
            }
          }
        }
      }  // for all pxls in window

      for0(k, nband)
        (c[k] = 0, w[k] = mx[k] - mn[k]); // w = metawindow width

      // put stuff into bins
      for(di = -dw; di <= dw; di++){
        
	dx = i + di; // row idx of pxl in wndw
        if(dx >= nrow || dx < 0) continue;
        
	ix = dx * ncol; // row contrib to data idx
        for(dj = -dw; dj <= dw; dj++){
          
	  dy = j + dj;  // col contrib to data idx
          if(dy >= ncol || dy < 0) continue;

          for0(k, nband){
            if(w[k] == 0.) continue;  // skip this band if no width

            dk = np * k;  // band contrib to data idx
            d = dat[ix + dy + dk];  // data value in window

            if(!(isinf(d) || isnan(d))){
 	      // bin index, this band
	      ci = (long int) (double)floor((double)n_bin * (d - mn[k]) / w[k]);
              
	      // increment count in bin
	      if(ci < 0 || ci > nbin)
	        err("invalid bin idx");
              else
                c[k * nbin + ci] += 1;
            }
          }  // for each band of pxl in wndw
        }  // for each col in wndw
      }  // for each row in wndw

      // assign a value for pixl (i,j) of output, cf greatest count. Don't test uniqueness
      for0(k, nband){
        xi = (np * k) + (i * ncol) + j;
        if(w[k] > 0){
          (max_i = 0, max_c = -FLT_MAX);

          for0(di, nbin)
            if(c[k * nbin + di] > max_c)
              (max_c = c[k * nbin + di], max_i = di);
          
	  out[xi] = (((float)max_i) + .5) * w[k] + mn[k];
        }
        else
          out[xi] = mn[k]; // box has same values in it.
      }  // for each band
    }  // for each col
  }  // for each row

  for0(k, np*nband) out[k];
  hwrite(out_hf, nrow, ncol, nband); // write output header
  str os("out.bin");
  bwrite(out, out_fn, nr, nc, nb);
  //free(dat);
  //free(out);
  return 0;
}
