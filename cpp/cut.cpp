/* cut rectangular subset from multispectral image etc. input assumed ENVI
type-4 32-bit IEEE standard floating-point format, BSQ interleave

similar to:
gdal_translate -srcwin xoff yoff xsize ysize -of ENVI -ot Float32 stack.bin stack_crop.bin

but does not keep header / geo info */

#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("cut [input binary file name] [startx] [starty] [endx] [endy] [output file name] ");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  str ofn(argv[6]);
  str ohfn(hdr_fn(ofn), create=true);
 
  printf("input file: %s\n", fn.c_str());
  printf("outputfile: %s\n", ofn.c_str());
  
  FILE * f_i = fopen(fn.c_str(), "rb");
  FILE * f_o = fopen(ofn.c_str(), "wb");

  size_t startx, starty, endx, endy;
  startx = (size_t) atol(argv[2]);
  starty = (size_t) atol(argv[3]);
  endx =   (size_t) atol(argv[4]);
  endy =   (size_t) atol(argv[5]);
  if(startx > endx || starty > endy) err("check subset coordinate params");

  size_t nrow, ncol, nband, np, i, j, k;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  size_t nrow2 = endy - starty + 1;
  size_t ncol2 = endx - startx + 1;
  size_t np2 = nrow2 * ncol2;

  float * bb = falloc(np); // buffer for one band
  float * bb2 = falloc(np2); // buffer for one band of output

  for0(k, nband){
    printf("band %zu of %zu\n", k + 1, nband);
    size_t fr = fread(bb, sizeof(float), np, f_i); // read a band
    if(fr != np) err("failed to read expected number of floats");
    
    for0(i, nrow2){
     size_t ix = i * ncol2;
     float * bb_start = &bb[((starty + i) * ncol) + startx];
      for0(j, ncol2) bb2[ix + j] = bb_start[j];
    }

    size_t fw = fwrite(bb2, sizeof(float), np2, f_o);
    if(fw != np2) err("failed to write expected number of floats"); 
  }

  hwrite(ohfn, nrow2, ncol2, nband); // write output header

  free(bb);  // clean up
  fclose(f_i);
  fclose(f_o);
  return 0;
}
