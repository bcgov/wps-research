/* 20211206: adapted from raster_negate.cpp
20220408: add sub-dominant version */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_dominant.exe [hyperspec cube] [optional arg: subdominant mode] # multiply by indicator fxn of dominant\n");

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  str ofn(str(argv[1]) + (argc < 3 ? str("_dominant.bin") : str("_subdominant.bin"))); // output file name: 1-hot encoding
  str ohn(hdr_fn(ofn, true)); // out header file name

  str oln(str(argv[1]) + (argc < 3 ? str("_dominant_label.bin") : str("_subdominant_label.bin"))); // output file, as label
  str olh(str(argv[1]) + (argc < 3 ? str("_dominant_label.hdr") : str("_subdominant_label.hdr")));

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix

  size_t n, i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband); // save as 1-hot
  float * dat = bread(fn, nrow, ncol, nband);
  float * lab = falloc(nrow * ncol); // save as label

  float dik; // data value
  float dom_v; // dominant value this pixel
  size_t dom_k = 0; // dominant index
  bool all_zero = true;

  for0(i, nrow){
    ix = (i * ncol);
    for0(j, ncol){
      ij = ix + j;

      ik = ij;
      dom_k = 0; //assume first band dominant
      dom_v = dat[ik];
      all_zero = true;

      for0(k, nband){
        ik = (np * k) + ij;
        dik = dat[ik];
        if(dik != 0.) all_zero = false;
        if((argc < 3 && dik > dom_v) || (argc > 2 && dik < dom_v)){
          //if(dat[ik] > dom_v)
          dom_v = dik; // find dominant band
          dom_k = k;
        }
      }
      lab[ij] = (float)dom_k; // save dominant as label

      for0(k, nband){
        ik = (np * k) + ij;
        out[ik] = 0.;
        if(dom_k == k){
          out[ik] = 1. ;
        }
        if(all_zero){
          out[ik] = NAN;
        }
        if(isnan(dat[ik])){
          out[ik] = NAN;
          break;
        }
      }
      // end of pixel based operation
    }
  }
  bwrite(out, ofn, nrow, ncol, nband);
  bwrite(lab, oln, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, nband);
  hwrite(olh, nrow, ncol, 1);
  str cmd(str("cp -v ") + hfn + str(" ") + ohn);
  cout << cmd << endl;
  system(cmd.c_str());
  free(out);
  free(dat);
  return 0;
}
