/* raster_sum.cpp: band math, add hyperspectral cubes together

20230523 added NAN-tolerant behaviour. Add all numbers except nan, if there are any. Otherwise assign NAN
20220808: should generalize this to > 2 input files
20220824: generalized to N input files */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4) err("raster_sum.exe [raster cube 1] [raster cube 2] [output cube]\n");
  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  size_t i, j, k, ix, ij, ik, m;
  float * out, * dat;

  int N = argc - 2; // one arg for exe name and one for output cube name
  str ofn(argv[argc - 1]); // output file name
  if(exists(ofn)) err("output file exists");
  str ohn(hdr_fn(ofn, true)); // out header file name
  cout << "N: " << N << endl;

  vector<str> fn, hfn;
  for0(i, N){
    fn.push_back(str(argv[i + 1]));
    if(!exists(fn[i])) err("failed to open input file");
    str hfn(hdr_fn(fn[i]));
    hread(hfn, nrow2, ncol2, nband2); // read header
    if(i > 0 && (nrow != nrow2 || ncol != ncol2 || nband != nband2)) err("image shape mismatch");
    else (nrow = nrow2), (ncol = ncol2), (nband = nband2);
  }

  np = nrow * ncol; // number of input pix
  out = falloc(np * nband);
  for0(i, np * nband) out[i] = 0.;
	bool all_nan;
	float d;

  for0(m, N){
    printf("+r %s\n", fn[m].c_str());
    dat = bread(fn[m], nrow, ncol, nband);
    for0(i, nrow){
      ix = i * ncol;
      for0(j, ncol){
        ij = ix + j;
				all_nan = true;
        for0(k, nband){
          ik = ij + k * np;
					d = dat[ik];
					if(!isnan(d)){
            out[ik] += dat[ik];
						all_nan = false;
					}
        }
				if(all_nan) out[ik] = NAN;
      }
    }
    free(dat);
  }

  printf("+w %s\n", ofn.c_str());
  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
