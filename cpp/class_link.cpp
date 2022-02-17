#include"misc.h"
#include<unordered_set>
#include<unordered_map>
/* 20220216 group nearby (non-zero) segs using a moving window: any two labels
 in same window get merged */

unordered_map<size_t, size_t> p; // disjoint-set forest / union-find

size_t find(size_t x){
  if(p[x] == x) return x;
  else{
    p[x] = find(p[x]); // path compression
    return p[x];
  }
}

void unite(size_t x, size_t y){
  x = find(x);
  y = find(y);
  if(x == y) return; // already in same set
  else p[y] = x; // make x parent of y
}

int main(int argc, char ** argv){
  if(argc < 3) err("class_link.exe [input file name] [window width] # windowed seg grouping, ike top hat");
  size_t d, np, i, j, k, n, ij, ii, jj, di, dj, *dat, *out, nrow, ncol, nband;
  int nwin = atoi(argv[2]);
  str fn(argv[1]); /* data */
  str ofn(fn + "_link.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));

  size_t d_type = hread(hfn, nrow, ncol, nband);
  if(d_type != 16) err("expected type-16 (size_t) image");
  if(nband != 1) err("expected 1-band image");
  np = nrow * ncol;

  out = (size_t *)(void *)alloc(np * sizeof(size_t));
  dat = (size_t *)(void *)alloc(np * sizeof(size_t));
  FILE * g = ropen(fn);
  size_t nr = fread(dat, nrow*ncol* sizeof(size_t), 1, g);

  cout << "expected size: " << nrow * ncol * sizeof(size_t) << endl;
  if(nr != 1){
    cout << "nr: " << nr << endl;
    err("incorrect read count");
  }

  for0(i, np){
    d = dat[i];
    p[d] = d;
  }
  unordered_set<size_t> merge;
  unordered_set<size_t>::iterator it;
  int frac = nwin / 2; // 2 could be another whole number
  for(i = 0; i < ncol + frac; i += frac){
    printf("%zu\n", i);
    for(j = 0; j < nrow + frac; j += frac){
      merge.clear();
      for0(di, nwin){
        ii = i + di;
        if(ii < nrow){
          for0(dj, nwin){
            jj = i + dj;
            if(jj < ncol){
	      d = dat[ii * ncol + jj];
              if(d > 0) merge.insert(d);
            }
          }
        }
      }
      if(merge.size() > 1){
        size_t parent = *(merge.begin());
        for(it = merge.begin(); it != merge.end(); it++){
          if(it != merge.begin()) unite(parent, *it);
        }
      }
    }
  }

  for0(i, nrow) for0(j, ncol){
    ij = i * ncol + j;
    d = dat[ij];
    out[ij] = (d == (size_t)0) ? (size_t)0 : find(d);
  }

  FILE * f = wopen(ofn);
  fwrite(out, sizeof(size_t), np, f);
  hwrite(hf2, nrow, ncol, 1, 16); /* type 16 = size_t */
  free(dat);
  free(out);
  return 0;
}
