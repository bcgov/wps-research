/* raster_difference.cpp: given a parameter N (bands per date)
assuming within-date bands are in order, difference a stack band by band :

Supposing M bands of N bands per date, there are M/N dates..

..then this program produces M/N - 1 output bands (after - before) */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3){
    err("raster_difference.exe [multidate cube] [# of bands per date]\n");
  }

  size_t M = atoi(argv[2]); // bands per date
  printf("bands per date: %zu\n", M);

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  str ofn(str(argv[1]) + str("_difference.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());
  
  size_t nrow, ncol, nband, np, n_dates;
  hread(hfn, nrow, ncol, nband); // read header 1
  n_dates = nband / M;

  np = nrow * ncol; // number of input pix
  size_t n, i, j, K, k1, k2, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat = bread(fn, nrow, ncol, nband);
  printf("number of dates: %zu\n", nband/M);

  for0(n, (n_dates - 1)){
    printf("di=%zu\n", n);
    // subtract date N+1 minus date N
    for0(K, M){
      // start idx of bands to subtract
      k1 = (n * M + K) * np;
      k2 = (M * np) + k1;

      // create one product band
      for0(i, nrow){
        ix = i * ncol;
        for0(j, ncol){
          ij = ix + j;
          out[k1 + ij] = dat[k2 + ij] - dat[k1 + ij];
        }
      }
    }
  }
  bwrite(out, ofn, nrow, ncol, nband - M);
  hwrite(ohn, nrow, ncol, nband - M);
  return 0;
}
