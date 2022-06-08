/* 20201107 deduplicate bip data. Note: Sorting cheaper than n* n. Use
as recursor for n* n calc? */
#include"misc.h"
#include<vector>
#include<algorithm>
#include<map>

float * dat; // input data bip format
size_t nr, nc, nb, nz;

class px{
  public: float * d;

  void init(size_t ix){
    d = &dat[ix * nb];
  }

  void check_data(){
    size_t k;
    for0(k, nb){
      float x = d[k];
      if(isnan(-x) || isinf(-x) || isnan(x) || isinf(x)) err("found nan or inf");
    }
    bool is_z = true;
    for0(k, nb) if(d[k] != 0.) is_z = false;
    if(is_z) nz ++;
  }
};

/* output operator for set container */
std::ostream& operator << (std::ostream& os, const px &v){
  size_t k;
  os << "{";
  for0(k, nb) os << (k > 0 ? " " : "") << v.d[k];
  os << "}";
  return os;
}

bool operator < (const px& a, const px& b){
  size_t k; // dictionary order
  char * c = (char *)(void *)a.d;
  char * d = (char *)(void *)b.d;
  size_t n_byte = nb * sizeof(float);

  for0(k, n_byte) if(c[k] != d[k]) return c[k] < d[k];
  return false;
}

int main(int argc, char ** argv){
  if(argc < 2){
    err("dedup [input file bip format] # deduplicate bip data");
  }

  nz = 0; // count zero vectors
  str inf(argv[1]); // input file
  str hfn(hdr_fn(inf)); // input header
  size_t np, i, j, ix, x;
  hread(hfn, nr, nc, nb); // read input hdr
  np = nr * nc; // number of pixels
  dat = bread(inf, nr, nc, nb); // read bip data

  px * p = new px[np];
  for0(i, np) p[i].init(i); // input
  map<px, size_t> m; // setoid index
  size_t * lookup = (size_t *) alloc(sizeof(size_t) * np);
  for0(i, np) lookup[i] = 0;

  size_t * s_i = (size_t *) alloc(sizeof(size_t) * np); // linear form of setoid index

  size_t next_i = 0;
  for0(i, nr){
    ix = i * nc;
    for0(j, nc){
      x = ix + j;
      px * pi = &p[x];
      pi->check_data(); // crash on nan/inf
      if(m.count(*pi) == 0){
        m[*pi] = x; // retain index to one element of setoid
        s_i[next_i] = x;
        lookup[x] = next_i ++;
      }
      else{
        cout << x << " redundant point: " << *pi << endl;
        lookup[x] = lookup[m[*pi]];
      }
    }
  }

  if(np < 10000){
    printf("lookup:\n");
    for0(i, np){
      printf("%zu -> %zu\n", i, lookup[i]);
    }
    printf("\n");
  }

  size_t ms = m.size();
  printf("data points total %zu\n", np);
  printf("number of setoids %zu\n", ms);
  printf("number of zeropts %zu\n", nz);
  printf("redundant points %zu\n", np - ms);

  i = 0;
  size_t * idx = (size_t *) alloc(sizeof(size_t) * m.size());
  map<px, size_t>::iterator it;
  for(it = m.begin(); it != m.end(); it++){
    size_t L_i = it->second;
    printf("%zu %zu %zu\n", i, L_i, s_i[i]);
    idx[i++] = s_i[i]; //L_i;
  }

  str ofn(inf + str("_dedup")); // input file

  FILE * f = wopen(ofn);
  size_t n_r = fwrite(idx, sizeof(size_t), m.size(), f);
  fclose(f);
  printf("records written %zu\n", n_r);
  if(n_r != m.size()) err("unexpected record write count");

  str of2(inf + str("_dedup_lookup"));
  f = wopen(of2);
  n_r = fwrite(lookup, sizeof(size_t), np, f);
  fclose(f);
  printf("records written %zu\n", n_r);
  if(n_r != np) err("unexpected record write count");

  free(idx);
  m.clear();
  return 0;
}
