/* squiggle.cpp: named suchly due to mental efficiency at computational
expense. Expand a raster with nb bands, for a result with: n * n * nb bands,
where n is the size of a window around a pixel. Out of bounds entries are
assigned window centre values! 

This adapter transforms moving-window into pixels, i.e. to run a pixel based
classifier using a moving window */

#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 3) err("squiggle [input raster] [window size (odd)]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  size_t np, nr, nc, nb, i, j, k, npk, ci;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * dat = bread(fn, nr, nc, nb); // read the input data
  int nwin = atoi(argv[2]); // square window length
  int dw = (nwin - 1) / 2; // window increment
  
  if((nwin - 1) % 2 != 0) err("window size must be odd");
  float * out = falloc(nr * nc * nb * nwin * nwin);
  str ofn(fn + str("_squiggle.bin"));
  str ohn(fn + str("_squiggle.hdr"));
  long int ii, jj;
  int di, dj;
  float d;
  ci = 0;

  for(di = -dw; di <= dw; di++){
    for(dj = -dw; dj <= dw; dj++){
      for0(k, nb){
	npk = np * k;

        for0(i, nr){
          ii = i + di;

          for0(j, nc){
            jj = j + dj;

            if(ii >= 0 && ii < nr && jj > 0 && jj < nc)
              out[ci++] = dat[npk + (ii * nc) + jj];
            else
              out[ci++] = dat[npk + (i * nc) + j];
          }
        }
      }
    }
  }
  bwrite(out, ofn, nr, nc, nb * nwin * nwin);
  hwrite(ohn, nr, nc, nb * nwin * nwin);
  return 0;
}
