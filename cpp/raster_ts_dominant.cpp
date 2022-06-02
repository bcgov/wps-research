/* 20220602: adapted from raster_dominant.cpp. This version assumes a multi-date (BSQ and organized by date/ in order, same number of bands per date)
20211206: adapted from raster_negate.cpp
20220408: add sub-dominant version */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("raster_dominant.exe [hyperspec cube] [bands per date]\n");
  size_t bands_per_date = (size_t)atol(argv[2]);
  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  str ofn(str(argv[1]) + str("_dominant.bin"));
  str ohn(hdr_fn(ofn, true)); // out header file name
  str oln(str(argv[1]) + str("_dominant_label.bin"));
  str olh(str(argv[1]) + str("_dominant_label.hdr"));

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix
  if(nband % bands_per_date != 0){
    err("number of bands must be multiple of bands_per_date");
  }
  size_t n_dates = nband / bands_per_date;
  size_t n, i, j, k, ix, ij, ik, m, date_offset;
  float * out = falloc(nrow * ncol * nband); // save as 1-hot
  float * dat = bread(fn, nrow, ncol, nband);
  float * lab = falloc(nrow * ncol * n_dates); // save as label

  float dik; // data value
  float dom_v; // dominant value this pixel
  size_t dom_k = 0; // dominant index
  bool all_zero = true;

  for0(m, n_dates){
    date_offset = np * bands_per_date * m;
    size_t lab_date_offset = np * m;

    for0(i, nrow){
      ix = (i * ncol);
      printf("date %zu row %zu n_dates %zu lab_date_off %zu labsize %zu\n", m, i, n_dates, lab_date_offset, nrow * ncol * n_dates);
      for0(j, ncol){
        ij = ix + j;

        ik = ij;
        dom_k = 0; //assume first band dominant
        dom_v = dat[ik + date_offset];
        all_zero = true;

        for0(k, bands_per_date){
          ik = date_offset + (np * k) + ij;
          dik = dat[ik];
          if(dik != 0.){
		  all_zero = false;
	  }
          if(dik > dom_v){
            dom_v = dik; // find dominant band
            dom_k = k;
          }
        }
        lab[ij + lab_date_offset] = (float)dom_k; // save dominant as label

        for0(k, bands_per_date){
          ik = date_offset + (np * k) + ij;
          out[ik] = 0.;
          if(dom_k == k){
            out[ik] = 1. ;
          }
          if(all_zero){
		  printf("ik %zu date_off %zu nrow *ncol*nband\n", ik, date_offset, nrow*ncol*nband); 
            out[ik] = NAN;
          }
        }
        // end of pixel based operation
      }
    }
  }
  bwrite(out, ofn, nrow, ncol, nband);
  bwrite(lab, oln, nrow, ncol, n_dates);
  hwrite(ohn, nrow, ncol, nband);
  hwrite(olh, nrow, ncol, n_dates);
  str cmd(str("cp -v ") + hfn + str(" ") + ohn);
  cout << cmd << endl;
  system(cmd.c_str());
  free(out);
  free(dat);
  return 0;
}
