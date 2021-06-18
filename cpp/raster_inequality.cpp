/* generalizing an idea found in dr. Dey's paper, to more dimensions.. */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("raster_inequality [input binary file name] [multilook factor] ");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ip, jp, ix1, ix2; // variables
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  n = (size_t) atoi(argv[2]);
  printf("multilook factor: %zu\n", n); // read mlk factor

  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  size_t nrow2 = nrow / n; // output image row dimensions
  size_t ncol2 = ncol / n;
  size_t np2 = nrow2 * ncol2; // allocate space for output
  size_t nf2 = np2 * nband;

  float * count = (float *) falloc(nf2);
  float * dat2 = (float *) falloc(nf2);
  for0(i, nf2) count[i] = dat2[i] = 0.; // set to zero

  for0(i, nrow){
    ip = i / n;
    for0(j, ncol){
      jp = j / n;
      for0(k, nband){
        ix1 = k * nrow * ncol + i * ncol + j;
        ix2 = k * nrow2 * ncol2 + ip * ncol2 + jp;
        float d = dat[ix1];
        if(ix2 < nf2 && !isnan(d) && !isinf(d)){
          dat2[ix2] += d;
          count[ix2]++;
        }
      }
    }
  }

  // divide by n
  for0(ip, nrow2){
    for0(jp, ncol2){
      for0(k, nband){
        ix1 = (k * np2) + (ip * ncol2) + jp;
        if(count[ix1] > 0.) dat2[ix1] /= count[ix1];
      }
    }
  }

  // write output file
  str ofn(fn + str("_mlk.bin"));
  str ohfn(fn + str("_mlk.hdr"));

  printf("nr2 %zu nc2 %zu nband %zu\n", nrow2, ncol2, nband);
  hwrite(ohfn, nrow2, ncol2, nband); // write output header

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  for0(i, nf2) fwrite(&dat2[i], sizeof(float), 1, f); // write data

  fclose(f);
  free(dat);
  free(dat2);
  free(count);
  return 0;
}
