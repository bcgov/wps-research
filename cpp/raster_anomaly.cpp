/* 20241205 raster_anomaly.cpp 

Generalization of sentinel2_anomaly2.cpp which was: "thermal" anomaly idea for sentinel2
   NB we didn't check if the headers of the two files match
*/
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 3){
    err("raster_anomaly [pre date sentinel-2] [post date sentinel-2]");
  }

  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat1, *dat2, * b11, * b21, * b31, *b12, *b22, *b32; 
  long int bi[3];

  str fn1(argv[1]); /* binary files */
  str fn2(argv[2]);
  str hfn(hdr_fn(fn1)); /* headers */

  str ofn(fn1 + "_" + fn2 + "_ratio.bin");
  str ohn(hdr_fn(ofn, true));

  vector<str> s; 
  vector<str> t;
  hread(hfn, nrow, ncol, nband, s);
  size_t nrow2, ncol2, nband2;

  hread(hdr_fn(fn2), nrow2, ncol2, nband2, t);
  
  if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
    err("image dimensions mismatch");
  }
  
  np = nrow * ncol;
  dat1 = bread(fn1, nrow, ncol, nband); /* read the data */
  dat2 = bread(fn2, nrow, ncol, nband); /* read the data */
  out = falloc(np * nband);

  for0(k, nband){
    j = k * np;
    for0(i, np){
      ij = j + i; 
      out[i] = (dat2[ij] - dat1[ij]) / (dat2[ij] + dat1[ij]);
    }
  }

  vector<str> bn;
  for0(i, nband){
    bn.push_back(str("(") + t[i] + str(" - ") + s[i] + str(") / (") + t[i] + str(" + ") + s[i] + str(")"));
  }
  hwrite(ohn, nrow, ncol, nband, 4, bn);
  bwrite(out, ofn, nrow, ncol, nband);
  run(str("envi_header_copy_mapinfo.py ") + hfn + str(" ") + ohn);  
  free(dat1);
  free(dat2);
  free(out);
  return 0;
}
