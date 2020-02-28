// extract_poly.cpp: extract average spectra from hyperspectral image

// inputs: 
// 1) multispectral image (ENVI type 4)
// 2) co-registered label map (ENVI type 4)

// outputs:
// 1) averaged product: same values over polygons
// 2) averages, by polygon index, for each polygon

int main(int argc, char ** argv){
  if(argc < 3) err("extract_poly [input binary file name] [input poly label file] [output binary file name] [output spectra file name]");

  str ifn(argv[1]); // input file name
  str hfn(hdr_fn(ifn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n;
  
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  float * dat = bread(fn, nrow, ncol, nband); 

  return 0;
}


