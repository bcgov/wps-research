#include"misc.h"
/* raster flood fill on mask: background is label 0: new labels to connected
comps of image areas equalling 1. 20220216 */

float * dat;
size_t * out;
size_t i_next, nrow, ncol, nband, nf;

void flood(long int i, long int j, size_t label){
  if(i < 0 || j < 0 || i >=nrow || j >=ncol) return;
  size_t ij = i * ncol + j;
  float d = dat[ij];

  if(d != 1. || out[ij] > 0) return; // labelled or not under mask
  out[ij] = label; // label this point
  nf ++; // marked something
  printf("i %zu j %zu\n", i, j);

  int di, dj;
  long int ii, jj;
  for(di = -1; di <= 1; di++){
    ii = i + di;
    for(dj = -1; dj <= 1; dj++){
      jj = j + dj;
      if(ii != i && jj != j) flood(ii, jj, label);
    }
  }
}

int main(int argc, char ** argv){
  if(argc < 2) err("flood.exe [input file name] # raster flood fill on mask");
  size_t np, i, j, k, n, ij;

  str fn(argv[1]); /* binary files */
  str ofn(fn + "_flood.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));

  hread(hfn, nrow, ncol, nband);
  np = nrow * ncol;
  if(nband != 1) err("expected 1-band image");

  out = (size_t *)(void *)alloc(np * sizeof(size_t));
  dat = bread(fn, nrow, ncol, nband);

  i_next = 1;
  for0(i, np) out[i] = -1.;
  for0(i, nrow) for0(j, ncol){
    nf = 0;
    flood(i, j, i_next);
    if(nf > 0) i_next ++;
  }

  FILE * f = wopen(ofn);
  fwrite(out, sizeof(size_t), np, f);
  hwrite(hf2, nrow, ncol, 1);
  free(dat);
  free(out);
  return 0;
}

/* list distances by passing a moving window. Then, merge any distances less than a threshold */
