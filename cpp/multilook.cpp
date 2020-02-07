/* multilook a multispectral image or radar stack, square window */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 3){
    err("multilook [input binary class file name] [multilook factor] ");
  }
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ip, jp;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  n = (size_t) atoi(argv[2]); // read multilook factor

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);

  // output image dimensions
  size_t nrow2 = nrow / n;
  size_t ncol2 = ncol / n;

  // allocate space for output
  size_t np2 = nrow2 * ncol2;
  size_t nf2 = np2 * nband;
  float * count = (float *) falloc(nf2);
  float * dat2 = (float *) falloc(nf2);

  for0(i, nf2){
    count[i] = dat2[i] = 0.;
  }

  for0(i, nrow){
    ip = i / n;
    for0(j, ncol){
      jp = j / n;
      for0(k, nband){
        size_t ix_1 = (k * nrow * ncol ) + ( i * ncol) + j;
        size_t ix_2 = (k * nrow2 * ncol2) + (ip * ncol2) + jp;
        float d = dat[ix_1];
        if(ix_2 < nf2 && !isnan(d) && !isinf(d)){
          dat2[ix_2] += d;
          count[ix_2] += 1.;
        }
      }
    }
  }

  // divide by n

  for0(ip, nrow2){
    for0(jp, ncol2){
      for0(k, nband){
        size_t ix = (k * nrow2 * ncol2) + (ip * ncol2) + jp;
        if(count[ix] > 0.){
          dat2[ix] /= count[ix];
        }
      }
    }
  }

  // write output file
  str ofn(fn + str("_mlk.bin"));
  str ohfn(fn + str("_mlk.hdr"));

  hwrite(ohfn, nrow2, ncol2, nband); // write output header
  cout << "+w " << ofn << endl;
  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");

  // write the inverted data
  cout << "nf2 " << nf2 << endl;
  cout << "nf2 * sizeof(f)" << nf2 * sizeof(float) << endl;
  for0(i, nf2){
    fwrite(&dat2, sizeof(float), 1, f);
  }
  fclose(f);

  free(dat);
  free(dat2);
  free(count);
  return 0;
}
