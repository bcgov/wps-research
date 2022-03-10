/* 20220309: scale each band of raster, to match the range of a different raster.
 
Note: input raster to transform, assumed already scaled in 0-1.

Note: will need to do histogram matching, to have meaningful comparison!! */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 3){
    err("raster_scale.exe [hyperspec cube to get max/min from] [hyperspec cube to transform (in 0-1 already)]\n");
  }

  str fn(argv[1]); // input image file name (to get max and min from)
  if(!(exists(fn))) err("failed to open input file #1");
  str hfn(hdr_fn(fn)); // input header file name

  str fn2(argv[2]);
  if(!(exists(fn2))) err("failed to open input file #2");
  str hfn2(hdr_fn(fn2)); // second input header file

  str ofn(fn2 + str("_scale_onto.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  size_t nrow, ncol, nband, nrow2, ncol2, nband2, np, i, j, k;
  hread(hfn, nrow, ncol, nband); // read header 1
  hread(hfn2, nrow2, ncol2, nband2);
  if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
    err("input images must have same shape");
  }
  np = nrow * ncol; // number of input pix
  float * dat = bread(fn, nrow, ncol, nband);
  float * dat2 = bread(fn2, nrow, ncol, nband);

  float * mn = falloc(nband); // min 
  float * mx = falloc(nband); // max 
  for0(k, nband){
    mn[k] = FLT_MAX;
    mx[k] = FLT_MIN;
  }

  float d;  // calculate min + max, each band
  for0(k, nband){
    j = k * np;
    for0(i, np){
      d = dat[j++];
      if(!(isnan(d) || isinf(d))){
        if(d < mn[k]) mn[k] = d;
        if(d > mx[k]) mx[k] = d;
      }
    }
  }

  for0(k, nband){
    printf(" %zu mn %f mx %f\n", k, mn[k], mx[k]);
  }

  for0(k, nband){
    float mk = mn[k];
    float m = (mx[k] - mk);
    for0(i, np){
      j = k * np + i;
      dat2[j] = (dat2[j] * m) + mk;
    }
  }
  bwrite(dat2, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
