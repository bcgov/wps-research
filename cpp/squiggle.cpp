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
  size_t np, nr, nc, nb, i, j, k;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * dat = bread(fn, nr, nc, nb); // read the input data
  int nwin = atoi(argv[2]); // square window length
  int dw = (nwin - 1) / 2; // window increment
  
  if((nwin - 1) % 2 != 0) err("window size must be odd");

  str ofn(fn + str("_squiggle.bin"));
  str ohn(fn + str("_squiggle.hdr"));
  FILE * f = wopen(ofn.c_str());
  long int ii, jj;
  int di, dj;
  float d;

  for(di = -dw; di <= dw; di++){
    for(dj = -dw; dj <= dw; dj++){
      for0(k, nb){
        for0(i, nr){
          ii = i + di;

          for0(j, nc){
            jj = j + dj;

            if(ii >= 0 && ii < nr && jj > 0 && jj < nc)
              d = dat[(np * k) + (ii * nc) + jj];
            else
              d = dat[(np * k) + (i * nc) + j];
            
            fwrite(&d, sizeof(float), 1, f);
          }
        }
      }
    }
  }
  fclose(f);
  hwrite(ohn, nr, nc, nb * nwin * nwin);
  return 0;
}
