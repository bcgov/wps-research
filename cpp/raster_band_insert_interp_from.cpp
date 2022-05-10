/* Assuming image is floating point 32bit IEEE standard type, BSQ interleave,
insert a single-band from one raster, into another:

modifying header accordingly
*/

#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 5){
    err("raster_band_insert [input BSQ image stack 1] [1-index for insertion] [input stack 2, containing desired band to be inserted] [1-index of band to be inserted] # optional args for interpolation transform: [1-index for left band, stack 1] [1-index for right band, stack 1] [1-index for left band, stack 2] [1-index for right band, stack 2] [wavelength for b1] [wavelength for b2] [wavelength for b3]");
  }

  float wl_1, wl_2, wl_3;
  int left1, left2, right1, right2;
  left1 = left2 = right1 = right2 = -1;
  if(argc == 12){
    left1 = atoi(argv[5]); right1 = atoi(argv[6]);
    left2 = atoi(argv[7]); right2 = atoi(argv[8]);
    wl_2 = atof(argv[10]); wl_3 = atof(argv[11]);
    wl_1 = atof(argv[9]);
  }
  float * bleft1, * bleft2, * bright1, * bright2;

  str fn(argv[1]); // input file name, stack which will get one band replaced
  str hfn(hdr_fn(fn)); // auto-detect header-file name

  str fn2(argv[3]); // input file name, including the single band that will be inserted into the first stack
  str hfn2(hdr_fn(fn2, true)); // auto-detect header-file name, don't assume exists

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  long int i, j, di, dj;
  hread(hfn, nrow, ncol, nband); // read header
  hread(hfn2, nrow2, ncol2, nband2); // read header, if exists 
  if(nrow != nrow2 || ncol != ncol2) err("please verify input image dimensions");

  int bi = atoi(argv[2]) - 1; // band index for inserted band to appear
  if(bi < 0 || bi > nband) err("please verify 1-index of band to splice");
  
  int bi2 = atoi(argv[4]) - 1; // band index of band to be spliced 
  if(bi2 < 0 || bi2 > nband2) err("please verify 1-index of band to be spliced");

  np = nrow * ncol;  // number of pixels per band

  // check sizes of both files, match headers
  size_t fs1 = fsize(fn);
  size_t fs2 = fsize(fn2);
  if(fs1 != np * nband * sizeof(float)) err("please verify file size for input stack");
  if(fs2 != np * nband2 * sizeof(float)) err("please verify file size for second input stack");
  // printf("file_size %zu\n", fs1);


  size_t f_p = np * sizeof(float) * (size_t) bi; // splice target location: should implement a shuffle version too
  size_t f_p2 = np * sizeof(float) * (size_t) bi2; // splice source location !!!!!!!

  FILE * f;

  // read in band to be spliced in (from 2nd file)
  float * bs = falloc(np); // alloc room to read a band
  f = fopen(argv[3], "rb");
  fseek(f, f_p2, SEEK_SET);
  fread(bs, sizeof(float), np, f);
  fclose(f);
 
  float b = wl_3 - wl_1;
  float w1 = (wl_2 - wl_1) / b;
  float w3 = (wl_3 - wl_2) / b;
  

  // interp mode:  do interp
  if(left1 > 0){
    // read stuff
    bleft1 = falloc(np); 
    bleft2 = falloc(np);
    bright1 = falloc(np);
    bright2 = falloc(np);
    
    f = fopen(argv[1], "rb");  // read "nextdoor" bands from first file
    fseek(f, (left1 - 1) * sizeof(float) * np, SEEK_SET);
    fread(bleft1, sizeof(float), np, f);
    fseek(f, (right1 - 1) * sizeof(float) * np, SEEK_SET);
    fread(bright1, sizeof(float), np, f);
    fclose(f);

    f = fopen(argv[3], "rb");  // read "nextdoor" bands from 2nd file
    fseek(f, (left2 - 1) * sizeof(float) * np, SEEK_SET);
    fread(bleft2, sizeof(float), np, f);
    fseek(f, (right2 - 1) * sizeof(float) * np, SEEK_SET);
    fread(bright2, sizeof(float), np, f);
    fclose(f);

    // now do the interp
    for0(i, np){
	    // calculate m 
	  /*
     bs[i] *= (w1 * (float)(log(  (exp((double)bleft2[i] + 1.) /  exp((double)bleft1[i] + 1.)) - 1.)) + 
	       w3 * (float)(log( (exp((double)bright2[i] + 1.) / exp((double)bright1[i] + 1.)) - 1.)));
    
     */
     bs[i] *= (w1 *  (bleft2[i]  / bleft1[i]) + 
               w3 *  (bright2[i] / bright1[i]));

    }
  }

  
  size_t to_buf = np * (1 + nband - bi);  // number of floats to buffer
  float * buf = falloc(to_buf);
  f = fopen(argv[1], "r+b");
  fseek(f, f_p, SEEK_SET);
  fread(buf, sizeof(float), to_buf, f);  // ok we buffered.
   
  // write insert band
  fseek(f, f_p, SEEK_SET);
  size_t nr = fwrite(bs, sizeof(float), np, f);

  nr += fwrite(buf, sizeof(float), to_buf, f);
  fclose(f);
 
  // qa: assert file size unchanged
  size_t fs1p = fsize(fn);
  if(fs1p != fs1 + (sizeof(float) * np)){
    printf("File size: %zu expected %zu\n", fs1p, fs1);
    err("splice failed");
  }
  return 0;
}
