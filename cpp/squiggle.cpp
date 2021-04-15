/* squiggle.cpp: named suchly because it is an efficient use of brain resource,
but not computer resource:

Take a raster and expand it such that: for every pixel with nb bands, there are now
n * n * nb bands where n is the size of a window around the pixel where:
  out of bounds entries are assigned the centre values..

That is, this adapter transforms moving-window data into pixel based..

i.e. can use it to run a pixel based classifier on a moving window */

#include"misc.h"
int main(int argc, char ** argv){
  
  if(argc < 3) err("squiggle.exe [input binary file name] [window size]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  size_t np, nr, nc, nb, i, j, k;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * dat = bread(fn, nr, nc, nb); // read the input data
  
  int nwin = atoi(argv[2]); // square window length
  int dw = (nwin - 1) / 2; // window increment
  if((nwin - 1) % 2 != 0) err("window size must be odd");

  str ofn(fn + str("squiggle.bin"));
  str ohn(fn + str("squiggle.hdr"));

  FILE * f = wopen(ofn.c_str());

  long int ii, jj;
  int di, dj;
  float d;

  for(di = -dw; di <= dw; di++){
    for(dj = -dw; dj <= dw; dj++){
      for0(k, nb){
        // for each band
        for0(i, nr){
          ii = i + di;
          for0(j, nc){
            jj = j + dj;
            if(ii >= 0 && ii < nr && jj > 0 && jj < nc){
              // in bounds
                d = dat[(np * k) + (ii * nc) + jj];
            }
            else{
              d = dat[(np * k) + (i * nc) + j];
            }
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

