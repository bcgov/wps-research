/* raster_sub.cpp: band math,
  for one cube:
  difference each combination of bands!  20220205  */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_sub_compare.exe [raster cube 1]\n");

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str ofn(fn + str("_sub_compare.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik, m, im, out_i;

  size_t nband2 = nband * nband; // number of output bands
  float * out = falloc(np * nband2);
  float * dat = bread(fn, nrow, ncol, nband);

  for0(i, np * nband2) out[i] = 0.;

  for0(i, nrow){
    if(i % 100 == 0) printf("%zu / %zu\n", i, nrow);
    ix = i * ncol;
    for0(j, ncol){
      ij = ix + j;
      for0(k, nband){
        ik = ij + k * np;

	for0(m, nband){
	  im = ij + m * np;
          out_i = k * nband + m;
	  out[ij + out_i * np] = dat[ik] - dat[im];
	}
      }
    }
  }

  bwrite(out, ofn, nrow, ncol, nband2);
  hwrite(ohn, nrow, ncol, nband2);
  return 0;
}
