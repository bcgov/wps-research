#include"misc.h"
/* raster flood fill on mask: background is label 0: new labels to connected
comps of image areas equalling 1. 20220216 */

float * dat;
int * visited;
size_t *out, i_next, nrow, ncol, nband, nf;
float * out_f;

void flood(long int i, long int j, size_t label){
  printf("i %ld j %ld\n", i, j);
  if(i < 0 || j < 0 || i >=nrow || j >=ncol) return;
  long int ij = i * ncol + j;
  float d = dat[ij];

  if(d != 1. || visited[ij]) return; // labelled or not under mask
  visited[ij] = true;
  out[ij] = label; // label this point
  nf ++; // marked something
  printf("\ti %ld j %ld label=%zu\n", i, j, label);

  long int di, dj;
  long int ii, jj;
  for(di = -1; di <= 1; di++){
    ii = i + di;
    for(dj = -1; dj <= 1; dj++){
      jj = j + dj;
      if(di != 0 && dj != 0){
	    if(!visited[ii*ncol + jj] && dat[ii*ncol + jj] !=0.)
	      flood(ii, jj, label);
      }
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

  str ofn4(fn + "_flood4.bin");
  str hfn4(fn + "_flood4.hdr");

  size_t d_type = hread(hfn, nrow, ncol, nband);
  np = nrow * ncol;
  if(nband != 1) err("expected 1-band image");
  if(d_type != 4) err("expected type-4 image");

  out = (size_t *)(void *)alloc(np * sizeof(size_t));
  dat = bread(fn, nrow, ncol, nband);
  visited = (int *)(void *)alloc(np * sizeof(int));
  out_f = falloc(np);

  i_next = 1;
  for0(i, np){
	  out[i] = 0;
	  visited[i] = false;
  }
  for0(i, nrow) for0(j, ncol){
    if(!visited[i * ncol + j]){
      nf = 0;
      flood((long int)i, (long int)j, i_next);
      if(nf > 0){
  	    cout << i << "," << j << "=" << i_next << endl;
  	    i_next ++;
      }
     }
  }
  for0(i, np) out_f[i] = (float)out[i];

  FILE * f = wopen(ofn);
  fwrite(out, sizeof(size_t), np, f);
  hwrite(hf2, nrow, ncol, 1, 16); /* type 16 = size_t */

  bwrite(out_f, ofn4, nrow, ncol, 1);
  hwrite(hfn4, nrow, ncol, 1, 4);
  free(visited);
  free(dat);
  free(out);
  return 0;
}
/* list distances by passing a moving window. Then, merge any distances less than a threshold */
