/* 20220605 convert PolSARPro complex format (one complex band envi type 6):
 to magnitude (r) and angle (theta) */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2){
    err("Convert PolSARPro complex format (envi type 6) to r, theta\nconvert_cplx_to_polar [input file]\n");
  }

  str fn(argv[1]);
  size_t i, j, nf, np, nr, nc, nb;
  float * d = float_read(fn, nf);
  if(nf % 2 != 0){
    free(d);
    err("number of floats in input file must be multiple of 2");
  }
  np = nf / 2;

  hread(hdr_fn(fn), nr, nc, nb);
  if(np != nr * nc) err("please check image dimensions");
  
  str of1(fn + str("_phase.bin"));
  str oh1(fn + str("_phase.hdr"));
  str of2(fn + str("_ampli.bin"));
  str oh2(fn + str("_ampli.hdr"));

  float * out1 = falloc(np);
  float * out2 = falloc(np);
  double re, im, r, theta;

  for0(i, np){
    j = 2 * i;
    (re = (double)d[j]), (im = (double)d[j + 1]);
    r = sqrt(re * re + im * im);
    theta = atan2(im, re);
    out1[i] = (float)theta;
    out2[i] = (float)r;
  }

  bwrite(out1, of1, nr, nc, 1);
  bwrite(out2, of2, nr, nc, 1);  
  hwrite(oh1, nr, nc, nb, 4);
  hwrite(oh2, nr, nc, nb, 4);
  free(out1);
  free(out2);
  free(d);
  return 0;
}
