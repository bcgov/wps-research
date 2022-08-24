/* raster_sum.cpp: band math, add hyperspectral cubes together

20220808: should generalize this to > 2 input files

20220824: generalize to N */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4) err("raster_sum.exe [raster cube 1] [raster cube 2] [output cube]\n");
  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  size_t i, j, k, ix, ij, ik, m;

  int N = argc - 2; // one arg for exe name and one for output cube name
  cout << "N " << N << endl;
  if(exists(argv[argc - 1])){
    err("output file exists");
  }

  vector<str> fn, hfn;
  for0(i, N){
    fn.push_back(str(argv[i + 1]));
    if(!exists(fn[i])) err("failed to open input file");
    str hfn(hdr_fn(fn[i]));
    hread(hfn, nrow2, ncol2, nband2); // read header
    if(i > 0 && (nrow != nrow2 || ncol != ncol2 || nband != nband2)) err("image shape mismatch");
    else (nrow = nrow2), (ncol = ncol2), (nband = nband2);
  }

  str ofn(argv[3]); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  np = nrow * ncol; // number of input pix
  float * out = falloc(nrow * ncol * nband);
  for0(i, np) out[i] = 0.;

  for0(m, N){
    float * dat = bread(fn[m], nrow, ncol, nband);
    for0(i, nrow){
      ix = i * ncol;
      for0(j, ncol){
        ij = ix + j;
        for0(k, nband){
          ik = ij + k * np;
          out[ik] += dat[ik];
        }
      }
    }
    free(dat);
  }

  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
