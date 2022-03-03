/* from a raster, generate a mask indicating where the raster satisfies the threshold e.g.:
cpp/raster_threshold.exe cloud.bin LEQ 10.
*/
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, i, j, k, ix, ij; // variables

  if(argc < 4){
    printf("From a raster, produce a mask that shows where the inequality is true:");
    err("raster_mask.exe [input binary file] [mode: GT, LT, GEQ, LEQ, EQ, NEQ] [threshold value]");
  }

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  set<str>modes;
  str mode(argv[2]);
  modes.insert(str("GT"));
  modes.insert(str("LT"));
  modes.insert(str("GEQ"));
  modes.insert(str("LEQ"));
  modes.insert(str("EQ"));
  modes.insert(str("NEQ"));
  if(modes.count(mode) < 1) err("mode string not found");

  float d;
  float thres = atof(argv[3]);
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(np * nband * sizeof(float));

  if(mode == str("GT")) for0(i, np) out[i] = (float)(dat[i] > thres);
  if(mode == str("LT")) for0(i, np) out[i] = (float)(dat[i] < thres);
  if(mode == str("GEQ")) for0(i, np) out[i] = (float)(dat[i] >= thres);
  if(mode == str("LEQ")) for0(i, np) out[i] = (float)(dat[i] <= thres);
  if(mode == str("EQ")) for0(i, np) out[i] = (float)(dat[i] == thres);
  if(mode == str("NEQ")) for0(i, np) out[i] = (float)(dat[i] != thres);

  str ofn(fn + str("_thres.bin")); // write output file
  str ohfn(fn + str("_thres.hdr"));
  hwrite(ohfn, nrow, ncol, nband); // write output header
  bwrite(out, ofn, nrow, ncol, nband);
  free(dat);
  free(out);
  return 0;
}
