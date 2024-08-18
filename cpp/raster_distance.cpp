/* 20240606 raster_distance.cpp: simple distance function between raster and:
user-supplied (vector) values
*/
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_distance [data cube]");
  }

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  size_t nrow, ncol, nband, np, n_dates;
  hread(hfn, nrow, ncol, nband); // read header 1

  printf("argc %d nband %zu\n", argc, nband);

  if(argc - 2 != nband){
    err(std::to_string(nband) + str(" parameter required"));
  }

  size_t i;

  float * x = falloc(nband);
  for0(i, nband){
    x[i] = std::stof(str(argv[i + 2])); // >> x[i];
  }

  str ofn(str(argv[1]) + str("_distance.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  np = nrow * ncol; // number of input pix
  size_t n, j, K, k1, k2, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat = bread(fn, nrow, ncol, nband);

  float d, di;
  for0(i, np){
    d = 0;
    for0(j, nband){
      di = dat[np * j + i] - x[j];
      d += di * di;
    }
    out[i] = -d;
  }
  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}
