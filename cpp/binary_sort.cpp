/* 20220516 sort a binary file, assuming it consists of BIP (float32) data */
#include"misc.h"
static size_t nb;

class p_ix{
  public: // float, index tuple object
  char * p;
  p_ix(char * q){
    p = q;
  }
  p_ix(const p_ix &a){
    p = a.p;
  }
};

bool operator<(const p_ix &a, const p_ix &b){
  size_t i; 
  for0(i, nb){
    if(a.p[i] < b.p[i]) return false;
  }
  return true;
}

int main(int argc, char ** argv){
  if(argc < 2) err("binary_sort [input raster]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  size_t np, nr, nc, i, k, ci;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  char * e = calloc(np * nb * sizeof(float));
  char * d = calloc(np * nb * sizeof(float));
  FILE * f = ropen(fn);
  size_t n_read = fread(d, 1, np * nb * sizeof(float), f);
  fclose(f);

  printf("alloc\n");
  char ** g = (char **)(void *)alloc(sizeof(char *) * np);
  for0(i, np) g[i] = NULL;

  priority_queue<p_ix> pq;
  for0(i, np){
    if(i % 10000 == 0) printf("push %zu / %zu\n", i, np);
    pq.push(p_ix(&d[i * nb * sizeof(float)]));
  }
  
  printf("pop\n");
  ci = 0;
  while(pq.size() > 0){
    if(ci % 10000 == 0)
      printf("pop  %zu / %zu\n", ci, np);
    p_ix x(pq.top());
    g[ci++] = x.p;
    pq.pop();
  }

  ci = 0;
  for0(i, np){
    for0(k, nb){
      e[ci++] = (g[i])[k];
    }
  }

  str ofn(fn + str("_sort.bin"));
  str ohn(fn + str("_sort.hdr"));
  
  // write output
  f = wopen(ofn);
  if(!f) err("failed to open outfile");
  size_t n_write = fwrite(e, 1, np * nb * sizeof(float), f);
  fclose(f);
  hwrite(ohn, nr, nc, nb);
  return 0;
}
