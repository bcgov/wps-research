/* 20220602 this version accepts a parameter "bands per date"..
..therefore, it accumulates multi-band results over time.

raster_accumulate.cpp: cumulative sum (by band) of single-band
raster sequence 20220320

20220429: need to update this to propagate source date/time from
band names strings, if available */
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, i, j, k, ii, ik, m, n_dates, date_offset;
  if(argc < 3) err("raster_ts_accumulate [raster cube] [bands per date] [optional arg:red only \n");
  int red_only = argc > 3;
  size_t bands_per_date = (size_t)atol(argv[2]);

  str fn(argv[1]); // input image
  str hfn(hdr_fn(fn)); // input header
  if(!exists(fn)) err("failed to open input file");

  str ofn(fn + str("_accumulate.bin")); // output image
  str ohn(hdr_fn(ofn, true)); // output header, create mode

  vector<str> band_names(parse_band_names(hfn)); // read band names
  hread(hfn, nrow, ncol, nband); // read dimensions from header
  np = nrow * ncol; // number of input pix
  n_dates = nband / bands_per_date;
  if(nband % bands_per_date != 0){
    err("nbands should be multiple of bands_per_date");
  }

  float * dat = bread(fn, nrow, ncol, nband); // read input
  float * out = falloc(nrow * ncol * nband); // output buffer
  size_t one_date_offset = np * bands_per_date;

  for0(m, n_dates){
    date_offset = np * bands_per_date * m;
    for0(k, bands_per_date){
      ik = (np * k) + date_offset;
      if(m == 0){
        for0(i, np){
          out[i + date_offset] = dat[i + date_offset]; //first band same
        }
      }
      else{
        for0(i, np){
          ii = i + ik;
          out[ii] = dat[ii] + out[ii - one_date_offset]; // add this band + last result
          if(out[ii] > 1.){
            out[ii] = 1.; // max result = 1.
          }
        }
      }
    }
  }

  hwrite(ohn, nrow, ncol, nband, 4, band_names); // write out dimensions to header
  bwrite(out, ofn, nrow, ncol, nband); // write out image
  return 0;
}
