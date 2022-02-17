#include"misc.h"
/* raster flood fill on mask: background is label 0: new labels to connected
 comps of image areas equalling 1. 20220216 */

float * dat;
size_t * out;
size_t next, nrow, ncol, nband, nf;

void flood(long int i, long int j, size_t label){
  if(i < 0 || j < 0 || i >=nrow || j >=ncol) return;
  size_t ij = i * ncol + j;
  float d = dat[ij];

  if(d != 1. || out[ij] > 0) return;


  /* execute this function at each point, with the data value */
  /* if the data value matches, keep going if we haven't been already */
}

int main(int argc, char ** argv){
  if(argc < 2) err("flood.exe [input file name] # raster flood fill on mask");
  size_t np, i, j, k, n, ij;

  str fn(argv[1]); /* binary files */
  str ofn(fn + "_flood.bin");
  str hfn(hdr_fn(fn));  /* headers */
  str hf2(hdr_fn(ofn, true));

  hread(hfn, nrow, ncol, nband);
  np = nrow * ncol;
  if(nband != 1) err("expected 1-band image");
  
  out = (size_t *)(void *)alloc(np * sizeof(size_t));
  dat = bread(fn, nrow, ncol, nband);

  next = 1;
  for0(i, np) out[i] = -1.;
  for0(i, nrow) for0(j, ncol){
	  nf = 0;
	  flood(i, j, next);
	  if(nf > 0) next ++;
  }

  FILE * f = wopen(ofn);
  fwrite(out, sizeof(size_t), np, f);
  hwrite(hf2, nrow, ncol, 1);
  free(dat);
  free(out);
  return 0;
}

/* list distances by passing a moving window. Then, merge any distances less than a threshold */
