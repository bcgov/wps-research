#include"misc.h"
/* raster flood fill on mask: background is label 0: new labels to connected
comps of image areas equalling 1. 20220216 */
float * dat;
int * visited;
size_t *out, i_next, nrow, ncol, nband, nf;
float * out_f;

int flood(long int i, long int j, int depth){
  printf("flood(%ld, %ld, %d) label=%ld d=%f depth=%d\n", i, j, depth, i_next, dat[i * ncol + j],depth);
  int ret = 0;
  if(i < 0 || j < 0 ||  i >= nrow || j >=ncol ){
    printf("oub\n");
    return 0;
  }
  long int ij = i * ncol + j;
  float d = dat[ij];

  if(d != 1. || visited[ij]){
	  printf("ret\n");
	  return 0; // labelled or not under mask
  }
  visited[ij] = true;
  out[ij] = i_next;
  nf ++; // marked something
//  printf("\ti %ld j %ld label=%zu depth=%d\n", i, j, i_next, depth);

  for(long int di = 1; di >= -1; di -= 1){
    long int ii = i + di;
    for(long int dj = 1; dj >= -1; dj -=1){
      long int jj = j + dj;
      printf("di %ld dj %ld (%ld, %ld) \n", di, dj, i, j);
      if((!(di == 0 && dj == 0)) &&
         (i + di >= 0) &&
	 (j + dj >= 0) &&
	 (i + di < nrow) &&
	 (j + dj < ncol)){

        long int ik = ii * ncol + jj;
        if(!visited[ik] && dat[ik] == 1.){
	  printf("  call(%ld, %ld) di %ld dj %ld\n", ii,jj, di, dj);
          ret += flood(ii, jj, depth+1);
	}
      }
    }
  }
  return ret;
}

int main(int argc, char ** argv){
  if(argc < 2) err("flood.exe [input file name] # raster flood fill on mask");
  size_t np, k, n;
  long int ij;
  long int i;
  long int j;

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
  for0(i, (long int)nrow){
    for0(j, (long int)ncol){
      ij = i * ncol + j;
      if(!visited[ij] && dat[ij] ==1.){
        nf = 0;
	cout << "--" << endl;
        int r = flood(i, j, 0);
        if(nf > 0){
          i_next ++;
        }
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
