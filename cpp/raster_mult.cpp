/* raster_mult.cpp: band math, multiply hyperspectral cubes together
adapted from raster_sum.cpp 20220302 */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4) err("raster_sum.exe [raster cube1] [raster cube2 (could be 1d)] [out cube]\n");
  str fn(argv[1]); // input image file name
  str fn2(argv[2]); // input image 2 file name
  if(!(exists(fn) && exists(fn2))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str hfn2(hdr_fn(fn2)); // input 2 header file name
  str ofn(argv[3]); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np, nf, i, k, nrow2, ncol2, nband2;
  hread(hfn, nrow, ncol, nband); // read header 1
  hread(hfn2, nrow2, ncol2, nband2); // read header 2
  if(nrow != nrow2 || ncol != ncol2 || (nband != nband2 && nband2 != 1))
  err("input image dimensions should match");

  np = nrow * ncol; // # input pix
  nf = np * nband; // # input float
  float * out = falloc(nf); // out buffer
  float * dat1 = bread(fn, nrow, ncol, nband);
  float * dat2 = bread(fn2, nrow, ncol, nband2);
  if(nband == nband2){
    for0(i, nf) out[i] = dat1[i] * dat2[i];
  }
  else{
    if(nband2 != 1) err("second image should have one band");
    for0(k, nband){
      for0(i, np){
        out[(k * np) + i] = dat1[(k * np) + i] * dat2[i];
      }
    }
  }
  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  cout << "nband " << nband << " nband2 " << nband2 << endl;
  return 0;
}
