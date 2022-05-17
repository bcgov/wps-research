/* 20220516 sort a binary file, assuming it consists of records of equal length N bytes */
#include"misc.h"
static size_t nb;

struct customLess{
  bool operator()(char * a, char * b) const {
    int j;
    for0(j, nb){
      if(a[j] < b[j]) return true;
    }
    return false;
  }
};

int main(int argc, char ** argv){
  if(argc < 2) err("binary_sort [input raster]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  size_t np, nr, nc, i, k, ci;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * dat = bread(fn, nr, nc, nb); // read the input data
  char * d = (char *)(void *)dat;
  float * out = falloc(np * nb);
  char * e = (char *)(void *)out;

  std::vector<char *> p;
  for0(i, np) p.push_back(&d[i * nb * sizeof(float)]);
  std::sort(p.begin(), p.end(), customLess());

  ci = 0;
  vector<char *>::iterator it;
  for0(k, nb){
    for(it = p.begin(); it != p.end(); it++)
      e[ci++] = (*it)[k];
  }

  str ofn(fn + str("_sort.bin"));
  str ohn(fn + str("_sort.hdr"));
  bwrite(out, ofn, nr, nc, nb);
  hwrite(ohn, nr, nc, nb);
  return 0;
}
