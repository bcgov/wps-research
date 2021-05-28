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

  size_t ws;
  int nbin;
  float n_bin;
  int dw;
  str fn("");
  str hfn("");

  if(argc < 4) {
    //err("mode_filter [input binary file] [window size] [number of bins]");
    ws = 3; //atoi(argv[2]);

    nbin = 2; //atoi(argv[3]);
    n_bin = (float)nbin; // bin size

    if(ws % 2 != 1) err("window size must be odd number");
    dw = (ws - 1) / 2;

    fn = str("rcm_0815_b002_rcm_0831_match_b002_rcm_0831_match_b002.bin_mlk.bin"); // input file name
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

  size_t nrow, ncol, nband, np, k, n;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  float d, * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(np * nband * sizeof(float));
  long int ix, iy, di, dj, dx, dy, dk, i, j;

  float * mn = falloc(nband); // window min this band
  float * mx = falloc(nband); // window max this band
  float * w = falloc(nband);
  float * c = falloc(nbin * nband); // count in "bin m" of for kth dimension at posn: k * nbin + m

  for0(i, nrow){
    for0(j, ncol){
      printf("i %d / %d j %d of %d\n", i, nrow, j, ncol);

      for0(k, nband){
        mn[k] = FLT_MAX;
        mx[k] = FLT_MIN;
      }

      //first calculate min, max
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
      for0(k, nband){
        c[k] = 0;
        w[k] = mx[k] - mn[k]; // metawindow length
        // cout << "w[k] " << w[k] << endl;
      }

      // now put stuff into bins
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
              int ci = (int) ( float)floor( n_bin * (d - mn[k]) / w[k]);
              if(ci < 0 || ci > nbin){
                cout << "ci: " << ci << endl;
                err("invalid ci");
              }
              int cki = k * nbin + ci;
              // cout << "cki: " << cki << endl;
              c[cki] += 1;
            }
          }
        }
      }
      //cout << " here \n";
      // assign a value based on the greatest count. Don't test uniqueness
      for0(k, nband){
        int xi = (np * k) + (i * ncol) + j;
        //cout << "xi: " << xi << " np*nb: " << np * nband << endl;
        if(w[k] > 0){
          int max_i = 0;
          float max_c = FLT_MIN;
          for0(di, nbin){
            if(c[k * nbin + di] > max_c){
              max_c = c[k * nbin + di];
              max_i = di;
            }
          }
          out[xi] = (((float)max_i) + .5) * w[k] + mn[k];
        }
        else{
          out[xi] = mn[k]; // box has same values in it.
        }
      }
    }
  }
  cout << "zing" << endl;
  cout << fn << endl;
  str ofn(str(fn) + str("_rmf.bin")); // write output file
  str ohfn(str(fn) + str("_rmf.hdr"));
  cout << ofn << endl;
  cout << "ohfn" << endl;
  cout << ohfn << endl;
  cout << "+w " << ohfn << endl;
  hwrite(ohfn, nrow, ncol, nband); // write output header
  cout << "+w " << ofn << endl;

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(out, sizeof(float), np * nband, f); // write data
  fclose(f);

  //free(dat);
  //free(out);
  return 0;
}