/* 20211206: raster_negate.cpp */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_dominant.exe [hyperspec cube] # multiply by indicator of dominant coordinate \n");
  }

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  str ofn(str(argv[1]) + str("_dominant.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix

  size_t n, i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat = bread(fn, nrow, ncol, nband);

  float dd; // dominant value this pixel
  size_t di = 0; // dominant index
  for0(i, nrow){
    ix = (i * ncol);
    for0(j, ncol){
      ij = ix + j;

      di = 0; //assume first band dominant
      ik = ij;
      dd = dat[ik];

      for0(k, nband){
	ik = (np * k) + ij;
	if(dat[ik] > dd){
	  dd = dat[ik];  // find dominant band
	  di = k;
	}
      }

      for0(k, nband){
        ik = (np * k) + ij;
	out[ik] = (di == k) ? dat[ik]: 0.;  // dominant band retains value
      }
    }
  }
  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
