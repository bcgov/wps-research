/* 20221106 reimplementation. Should set number of bins per band (manually)?
Or set number of bins per band (automatically) ? */
#include"misc.h"
int main(int argc, char ** argv){
  int n_bins;
  if(argc < 4){
    printf("modf.cpp: Band-wise Histogram binning mode filter on binary single-precision floating point file\n");
    printf("By A. Richardson, 20090821 revised 20170703 reimplemented 20221116\n");
    err("\n\tUsage: modf [input file] [Number of Histogram bins] [window size] [ optional ratio parameter, for what?\n");
  }
  size_t nrow, ncol, nband;
  register int i, j;
  float ratio = 1.;
  if(argc > 4) ratio = atof(argv[4]);
  n_bins = atoi(argv[2]);

  str ifn(argv[1]);
  str ofn(ifn + str("_modf.bin"));
  str ohfn(ifn + str("_modf.hdr"));
  str hfn(hdr_fn(ifn)); // auto-detect header file name
  hread(hfn, nrow, ncol, nband); // read header
  size_t np = nrow * ncol; // number of input pix

  printf("here\n");
  int windowsize = atoi(argv[3]);
  if(((windowsize - 1) < 2) || (((windowsize - 1) % 2) != 0)){
    err("incorrect window size. Must be an odd number.\n");
  }
  int dw = (windowsize - 1) / 2;

  float *data, *outd, *dat, *out;
  data = bread(ifn, nrow, ncol, nband); // load floats to array
  outd = falloc(np * nband); // output data

  float fmax, fmin, d;
  double total, total_squared_dif, dif;
  fmax = fmin = total = total_squared_dif = dif = 0.;
  
  int * hist = ialloc(n_bins);
  int idx = 0;

  int totaln = 0;
  size_t row, col;
  int ind, max, maxi;
  max = maxi = 0;

  int band, q;
  for0(band, nband){
    dat = &data[np * band];
    out = &outd[np * band];
    for(row = 0; row < nrow; row++){
      printf("\rProcessing row %zu of %zu (band %d)", row + 1, nrow, band);

      for(col = 0; col < ncol; col++){
        /* pixel window */
        totaln = total = 0.;
        fmax = -FLT_MAX;  // bugfix 
        fmin = FLT_MAX;
        for0(q, n_bins) hist[q] = 0.;

        for(i = row - dw; i <= row + dw; i++){
          for(j = col - dw; j < col + dw; j++){
            if((i >= 0) && (j >= 0) && (i < nrow) && (j < ncol)){
              ind = i * ncol + j;
              d = dat[ind];
              if(!(isinf(d) || isnan(d))){
                idx = (int)floor(d * ((float)n_bins));
                if(idx == n_bins){
                  idx = idx - 1;
                }
                hist[idx]++;
              }
            }
          }
        }
        max = maxi = 0;  // find largest bin
        for(i = 0; i < n_bins; i++){
          if(hist[i] > max){
            (max = hist[i]), (maxi = i);
          }
        }
	out[row * ncol + col] = ((float)maxi)/((float)n_bins);
      }
    }
  }
  printf("\r\n");
  printf("nr %zu nc %zu nband %zu\n", nrow, ncol, nband);
  hwrite(ohfn, nrow, ncol, nband); // write output header
  bwrite(outd, ofn, nrow, ncol, nband);
  free(hist);
  return 0;
}
