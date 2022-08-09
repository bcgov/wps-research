/* raster_sum.cpp: band math, add hyperspectral cubes together

20220808: should generalize this to > 2 input files */
#
include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4) err("raster_sum.exe [raster cube 1] [raster cube 2] [output cube]\n");

  str fn(argv[1]); // input image file name
  str fn2(argv[2]); // input image 2 file name
  if(!(exists(fn) && exists(fn2))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str hfn2(hdr_fn(fn2)); // input 2 header file name
  str ofn(argv[3]); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  hread(hfn, nrow, ncol, nband); // read header 1
  hread(hfn2, nrow2, ncol2, nband2); // read header 2
  if(nrow != nrow2 || ncol != ncol2 || nband != nband2)
    err("input image dimensions should match");

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat1 = bread(fn, nrow, ncol, nband);
  float * dat2 = bread(fn2, nrow, ncol, nband);

  for0(i, nrow){
    ix = i * ncol;
    for0(j, ncol){
      ij = ix + j;
      for0(k, nband){
        ik = ij + k * np;
        out[ik] = dat1[ik] + dat2[ik];
      }
    }
  }

  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
