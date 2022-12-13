/* 20221213: trace edges of segments. Classes in floating point format */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("class_edges [input binary file name]");
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name

  long int i, j, np, di, dj, ii, jj;
  hread(hfn, nrow, ncol, nband); // read header
  if(nband != 1) err("class map expected: one band");
  np = nrow * ncol; // number of input pix

  dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(np);
  for0(i, np) out[i] = 0.;
  float d;
  bool other;

  for0(i, nrow){
    if(i % 100 == 0) cout << i << " of " << (nrow + 1) << endl;
    ix = i * ncol;

    for0(j, ncol){
      (ij = ix + j), (other=false);
      d = dat[ij];  // class map value, this location

      for(di = -1; di <= 1; di++){
        ii = i + di;
        if(ii < 0 || ii >= nrow) continue;

        for(dj = -1; dj <= 1; dj++){
          if(di == 0 && dj == 0) continue;

          jj = j + dj;
          if(jj < 0 || jj >= ncol) continue;

          if(dat[ii * ncol + jj] != d) other = true;
        }
      }
      if(other) out[ij] = 1.;
    }
  }

  str ofn(fn + str("_edges.bin")); // output file writing
  str ohfn(fn + str("_edges.hdr"));
  hwrite(ohfn, nrow, ncol, 1); // write header
  bwrite(out, ofn, nrow, ncol, 1);
  return 0;
}
