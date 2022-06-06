/* multilook a multispectral image or radar stack, square or rectangular window, input
assumed ENVI type-4 32-bit IEEE standard floating-point format, BSQ
interleave

20220506 consider all-zero (if nbands > 1) equivalent to NAN / no-data */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("multilook [input binary file name] [vertical or square multilook factor] # [optional: horiz multilook factor]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  
  int zero;
  float d, *dat, *dat2, *count;
  size_t nrow, ncol, nband, np, i, j, k, n, m, ix1;
  size_t nrow2, ncol2, np2, ip, jp, ix2, ix_i, ix_ip, nf2;
  
  vector<str> band_names(parse_band_names(hfn));
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  n = (size_t) atol(argv[2]); // multilook factor
  m = n;  // assume proportional unless specified
  if(argc > 3) m = (size_t)atol(argv[3]);
  printf("multilook factor (row): %zu\n", n);
  printf("multilook factor (col): %zu\n", m);
  
  dat = bread(fn, nrow, ncol, nband);
  (nrow2 = nrow / n), (ncol2 = ncol / m);
  np2 = nrow2 * ncol2;
  nf2 = np2 * nband; 

  zero = true;
  dat2 = (float *) falloc(nf2);
  count = (float *) falloc(nf2);
  for0(i, nf2) count[i] = dat2[i] = 0.;

  for0(i, nrow){
    ip = i / n;
    for0(j, ncol){
      (jp = j / m), (ix_i = (i * ncol) + j), zero = true;
      for0(k, nband){
        if(dat[k * np + ix_i] != 0.){
          zero = false;
        }
      }
      if(zero && nband > 1){
      }
      else{
        ix_ip = (ip * ncol2) + jp;
        for0(k, nband){
          (ix1 = (k * np) + ix_i), (ix2 = (k * np2) + ix_ip);
          d = dat[ix1];
          if(ix2 < nf2 && !isnan(d) && !isinf(d)){
            dat2[ix2] += d;
            count[ix2]++;
          }
        }
      }
    }
  }

  for0(ip, nrow2){
    if(ip % 1000 == 0) printf("row %zu of %zu\n", ip+1, nrow2);
    
    for0(jp, ncol2){
      for0(k, nband){
        ix2 = (k * np2) + (ip * ncol2) + jp;  // divide by n
        dat2[ix2] = (count[ix2] > 0.)? (dat2[ix2] / count[ix2]): NAN;
      }
    }
  }

  str ofn(fn + str("_mlk.bin")); // write output file
  str ohfn(fn + str("_mlk.hdr"));
  printf("nr2 %zu nc2 %zu nband %zu\n", nrow2, ncol2, nband);
  hwrite(ohfn, nrow2, ncol2, nband, 4, band_names); // write header
  bwrite(dat2, ofn, nrow2, ncol2, nband); // write output

  free(dat);
  free(dat2);
  free(count);
  return 0;
}
