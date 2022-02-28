#include"misc.h"
#include<unordered_set>
#include<unordered_map>
/* 20220216 group nearby (non-zero) segs using a moving window: any two labels
in same window get merged */

unordered_map<float, set<size_t>> members;
unordered_map<float, float> p; //<size_t, size_t> p; // disjoint-set forest / union-find
set<str> merges;

float find(float x){
  if(p[x] == x){
    return x;
  }
  else{
    p[x] = find(p[x]); // path compression
    return p[x];
  }
}

bool unite(float x, float y){
  x = find(x);
  y = find(y);
  if(x == y){
    return false; // already in same set
  }
  else{
    if(x < y) p[y] = x; // make x parent of y
    else p[x] = y;
    return true;
  }
}

int main(int argc, char ** argv){
  if(argc < 3){
    err("class_link.exe [input file name] [window width] # windowed seg grouping, ike top hat");
  }
  size_t d, np, k, n, ij, nrow, ncol, nband;
  float * dat, * out;
  long int nwin = (long int)atoi(argv[2]);
  cout << "nwin " << nwin << endl;
  str fn(argv[1]); str ofn(fn + "_link.bin");
  str hfn(hdr_fn(fn)); str hf2(hdr_fn(ofn, true));
  long int i, j, di, dj, ii, jj;
  int debug = false;

  size_t d_type = hread(hfn, nrow, ncol, nband);
  if(d_type != 4) err("expected type-4 (float) image");
  if(nband != 1) err("expected 1-band image");
  np = nrow * ncol;
  out = falloc(np);
  dat = bread(fn, nrow, ncol, nband);

  for0(i, np){
    d = dat[i];
    if(d > 0){
      p[d] = d;
      if(members.count(d) < 1){
        members[d] = set<size_t>();
      }
      members[d].insert(i);
    }
  }

  for(unordered_map<float, set<size_t>>::iterator it = members.begin(); it != members.end(); it++){
    cout << endl;
    cout << it->first;
    cout << "={";
    for(set<size_t>::iterator ti = (*it).second.begin(); ti != (*it).second.end(); ti++){
      cout << *ti << ",";
    }
    cout << "}" << endl;
  }

  size_t iter = 0;
  unordered_set<float> merge;
  unordered_set<float>::iterator it;
  long int frac = nwin / 2; // 2 could be another whole number
  for(i = 0; i < nrow + frac; i += frac){
    for(j = 0; j < ncol + frac; j += frac){
      // printf("i %ld j %ld\n", i, j);
      merge.clear();

      for0(di, nwin){
        ii = i + di;
        if(ii < nrow){

          for0(dj, nwin){
            jj = j + dj;
            if(jj < ncol){

              d = dat[ii * ncol + jj];
              if(d > 0){
                // printf(" %ld %ld >0 %zu\n", ii, jj, d);
                merge.insert(d);
              }
            }
          }
        }
      }
      if(merge.size() > 1){
        float parent = *(merge.begin());
        for(it = merge.begin(); it != merge.end(); it++){
          if(it != merge.begin()){
            bool new_merge = unite(parent, *it);
            if(new_merge)
            merges.insert(to_string(parent) + str(",") + to_string(*it));
          }
        }
        cout << "iter" << iter << " merge: i " << i << " j " << j << merge << endl;
        // optional: write provisinal output this step
        if(debug){
          str ofn_i(str("merge_") + to_string(iter) + str(".bin"));
          str hfn_i(str("merge_") + to_string(iter) + str(".hdr"));
          for0(ii, nrow) for0(jj, ncol){
            ij = ii * ncol + jj;
            d = dat[ij];
            out[ij] = (d == (size_t)0) ? (size_t)0 : find(d);
          }

          cout << merges << endl;
          FILE * f = wopen(ofn_i);
          fwrite(out, sizeof(float), np, f);
          hwrite(hfn_i, nrow, ncol, 1, 4); /* type 16 = size_t */
        }
        iter ++;
      }
    }
  }

  for0(i, nrow) for0(j, ncol){
    ij = i * ncol + j;
    d = dat[ij];
    out[ij] = (d == (size_t)0) ? (size_t)0 : find(d);
  }

  cout << merges << endl;
  FILE * f = wopen(ofn);
  fwrite(out, sizeof(float), np, f);
  hwrite(hf2, nrow, ncol, 1, 4); /* type 16 = size_t */

  free(dat);
  free(out);
  return 0;
}
