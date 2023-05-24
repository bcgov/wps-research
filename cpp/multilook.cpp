/* multilook a multispectral image or radar stack, square or rectangular window, input
assumed ENVI type-4 32-bit IEEE standard floating-point format, BSQ
interleave
20230524 band by band processing to support larger file
20220506 consider all-zero (if nbands > 1) equivalent to NAN / no-data */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("multilook [input binary file name] [vertical or square multilook factor] # [optional: horiz multilook factor]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  str ofn(fn + str("_mlk.bin")); // write output file
  str ohfn(fn + str("_mlk.hdr"));

  int zero;
  float d, *dat, *dat2, *count;
  size_t nrow, ncol, nband, np, i, j, k, n, m, ix1;
  size_t nrow2, ncol2, np2, ip, jp, ix2, ix_i, ix_ip, nf2;

  vector<str> band_names(parse_band_names(hfn));
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  n = (size_t) atol(argv[2]); // multilook factor
  m = n; // assume proportional unless specified
  if(argc > 3) m = (size_t)atol(argv[3]);
  printf("multilook factor (row): %zu\n", n);
  printf("multilook factor (col): %zu\n", m);

  FILE * f = fopen(fn.c_str(), "rb");
  FILE * g = fopen(ofn.c_str(), "wb");
  dat = falloc(np);
  (nrow2 = nrow / n), (ncol2 = ncol / m);
  np2 = nrow2 * ncol2;
  nf2 = np2;

  zero = true;
  dat2 = (float *) falloc(nf2);
  count = (float *) falloc(nf2);
  for0(i, nf2) count[i] = dat2[i] = 0.;

  // add up elements
  for0(k, nband){
    printf("processing band %zu of %zu\n", k + 1, nband);
		printf("fread\n");
    size_t nr = fread(dat, np, sizeof(float), f);
    for0(i, nrow){
			if(i % 1000 == 0) printf("  row %zu of %zu\n", i, nrow);
      ip = i / n;
      for0(j, ncol){
        (jp = j / m), (ix_i = (i * ncol) + j), zero = true;
        ix_ip = (ip * ncol2) + jp;
        d = dat[ix_i];
        if(ix_ip < nf2 && !isnan(d) && !isinf(d)){
          dat2[ix_ip] += d;
          count[ix_ip]++;
        }
      }
    }

    // divide by n
    for0(ip, nrow2){
      for0(jp, ncol2){
        ix2 = (ip * ncol2) + jp; // divide by n
        dat2[ix2] = (count[ix2] > 0.)? (dat2[ix2] / count[ix2]): NAN;
      }
    }

    // write output band
		printf("fwrite\n");
    size_t nw = fwrite(dat2, np2, sizeof(float), g);
  }
  printf("nr2 %zu nc2 %zu nband %zu\n", nrow2, ncol2, nband);
  hwrite(ohfn, nrow2, ncol2, nband, 4, band_names); // write header

  fclose(f);
  fclose(g);
  free(dat);
  free(dat2);
  free(count);
  return 0;
}
