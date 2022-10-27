/* 20220309: scale each band of raster between 0 and 1 */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_scale.exe [hyperspec cube] # scale each band into 0-1\n");
  }

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  str ofn(str(argv[1]) + str("_scale.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  size_t nrow, ncol, nband, np, i, j, k;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix

  float * dat = bread(fn, nrow, ncol, nband);
  float * mn = falloc(nband);
  float * mx = falloc(nband);
  for0(k, nband){
    mn[k] = FLT_MAX;
    mx[k] = -FLT_MAX;
  }

  float d;
  for0(k, nband){
    // calculate min and max for each band
    j = k * np;
    for0(i, np){
      d = dat[j++];
      if(!(isnan(d) || isinf(d))){
        if(d < mn[k]) mn[k] = d;
        if(d > mx[k]) mx[k] = d;
      }
    }
  }

  for0(k, nband) printf(" %zu mn %f mx %f\n", k, mn[k], mx[k]);

  for0(k, nband){
    float mk = mn[k];
    float m = 1. / (mx[k] - mk);
    for0(i, np){
      j = k * np + i;
      dat[j] = (dat[j] - mk) * m;
    }
  }
  bwrite(dat, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
