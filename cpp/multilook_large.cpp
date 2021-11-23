/* multilook a multispectral image or radar stack, square window, input
assumed ENVI type-4 32-bit IEEE standard floating-point format, BSQ
interleave:

this version doesn't read in the whole image. Band by band */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("multilook [input binary file name] [multilook factor] ");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ip, jp, ix1, ix2; // variables
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  n = (size_t) atoi(argv[2]);
  printf("multilook factor: %zu\n", n); // read mlk factor

  /* parameters on new image */
  size_t nrow2 = nrow / n; // output image row dimensions
  size_t ncol2 = ncol / n;
  size_t np2 = nrow2 * ncol2; // allocate space for output
  float * count = (float *) falloc(np2);
  float * dat2 = (float *) falloc(np2); // dat for band to write

  /* define output files, write header */
  str ofn(fn + str("_mlk.bin"));
  str ohfn(fn + str("_mlk.hdr"));
  printf("nr2 %zu nc2 %zu nband %zu\n", nrow2, ncol2, nband);
  hwrite(ohfn, nrow2, ncol2, nband); // write output hdr

  /* init data buffer, open files */
  float * dat = (float *)falloc(np); // read one band at a time
  FILE * f = ropen(argv[1]); // open file for reading
  FILE * g = wopen(ofn.c_str()); // open output file to write

  for0(k, nband){
    size_t nr = fread(dat, sizeof(float), np, f); // read a band
    if(nr != np){
      printf("Err: number of floats read: %zu expected %zu\n", nr, np);
      err("read error");
    }

    for0(i, np2) count[i] = dat2[i] = 0.; // set to zero

    for0(i, nrow){
      ip = i / n;
      for0(j, ncol){
        jp = j / n;

        ix1 = /*k * nrow * ncol + */ i * ncol + j;
        ix2 = /*k * nrow2 * ncol2 + */ ip * ncol2 + jp;
        float d = dat[ix1];
        if(ix2 < np2 && !isnan(d) && !isinf(d)){
          dat2[ix2] += d;
          count[ix2]++;
        }
      }
    }

    // divide by n
    for0(ip, nrow2){
      for0(jp, ncol2){
        ix1 = /* (k * np2) */ (ip * ncol2) + jp;
        if(count[ix1] > 0.) dat2[ix1] /= count[ix1];
      }
    }

    size_t bw = fwrite(dat2, np2, sizeof(float), g);
    if(bw != np2){
      printf("%zu %zu\n", bw, np2);
      err("unexpected write count");
    }
  }
  fclose(f);
  free(dat);
  free(dat2);
  free(count);
  return 0;
}
