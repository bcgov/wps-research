/* from a raster, generate a mask indicating where the raster satisfies the threshold */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4){
    printf("From a raster, produce a mask that shows where the inequality is true:");
    err("raster_mask.exe [input binary file] [mode: GT, LT, GEQ, LEQ] [threshold value]");
  }
  exit(1);
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ip, jp, ix1, ix2; // variables
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
/*
  set<float> values;
  for0(i, number_of_values) values.insert(atof(argv[i + 2]));
  cout << "values : " << values << endl;

  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(nrow * ncol * sizeof(float));

  bool found;
  size_t ix;
  float d;
  set<float>::iterator it;
  for0(i, nrow){
    for0(j, ncol){
      ix = i * ncol + j;
      out[ix] = 0.;
      d = dat[ix];
      if(!(isnan(d) || isinf(d))){
        if(values.find(d) != values.end()) out[ix] = true;
      }
    }
  }

  str ofn(fn + str("_mask.bin")); // write output file
  str ohfn(fn + str("_mask.hdr"));
  hwrite(ohfn, nrow, ncol, nband); // write output header

  cout << "+w " << ofn << endl;
  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(out, sizeof(float) * nrow * ncol, 1, f); // write data

  fclose(f);
  free(dat);
  free(out);
 
 */
  return 0;

 }
