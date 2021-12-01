/* csv_spectra_raster_distance.cpp:
calculate the mean and stdv spectra for the class indicated

*/
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 5) err(str("csv_spectra_raster_distance ") +
                   str("[csv file] ") +  // csv file to get categorized spectra from
                   str("[col-label of field of interest] ") + // label of column containing categorical field
                   str("[observation of field of interest to match]") + // value of the categorical field, over which to average spectra
                   str("[raster image to project onto]"));
  str csv_fn(argv[1]); // input image file name
  str fn(argv[4]); // input image 2 file name
  if(!(exists(csv_fn) && exists(fn))) err("failed to open all input files");
  str hfn(hdr_fn(fn)); // input raster header file name
  
  // str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * dat = bread(fn, nrow, ncol, nband);


  //bwrite(out, ofn, nrow, ncol, nband);
  //hwrite(ohn, nrow, ncol, nband);
  return 0;
}
