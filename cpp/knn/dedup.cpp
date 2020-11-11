/* preconditioning for patch data. Null pixels within a patch are assigned the average of non-null pixels within a patch, if non-zero */
#include"../misc.h"

float * dat;
size_t nr, nc, nb, nf, np, ps, fpp, nz; // image dimensions, etc.

class px{
  public:
  float * d;
  void init(size_t ix){
    float f;
    bool bad;
    size_t i;
    d = &dat[ix * fpp];
    for0(i, fpp){
      f = d[i];
      bad = false;
      // should check if isinf(-d) == isinf(d)
      if(isinf(-f) || isnan(-f) || isinf(f) || isnan(f)){
        bad = true;
        break;
      }
      if(bad){
        printf("warning: bad data ix %zu\n", ix * fpp);
        for0(i, fpp) d[i] = 0.;
      }
    }
    bool zp = true;
    for0(i, fpp) if(d[i] != 0.) zp = false;
    if(zp) nz += 1;

  }
};

// output operator for patch container
std::ostream& operator << (std::ostream& os, const px &v){
  size_t k;
  os << "{";
  for0(k, fpp) os << (k > 0 ? " " : "") << v.d[k];
  os << "}";
  return os;
}

bool operator < (const px& a, const px& b){
  size_t k; // dictionary order
  char * c = (char *)(void *)a.d;
  char * d = (char *)(void *)b.d;
  size_t bpp = fpp * sizeof(float); // bytes per patch
  for0(k, bpp) if(c[k] != d[k]) return c[k] < d[k];
  return false;
}

int main(int argc, char ** argv){
  if(argc < 2) err("dedup [input fp stack .bin name, already patched]");

  size_t i, j, k;
  str bfn(argv[1]);
  str hfn(hdr_fn(bfn));
  hread(hfn, nr, nc, nb);
  printf("nr %zu nc %zu nb %zu\n", nr, nc, nb);
  nb = int_read(bfn + str("_nb")); // number of actual image data bands (not incl. groundref)
  ps = int_read(bfn + str("_ps")); // patch size
  fpp = ps * ps * nb; // number of floats per patch

  str pfn(bfn + str("_patch"));
  nf = fsize(pfn) / sizeof(float); //number of floats
  np = nf / fpp; // number of patches

  printf("patch size: %zu\n", ps); // patch size
  printf("floats per patch: %zu\n", fpp);
  printf("number of patches: %zu\n", np);

  dat = float_read(pfn, nf); // read patch data, as linear array of floats
  if(nf != np * fpp) err("unexpected number of floats read");

  // printf("precondition..\n");
  px * p = new px[np];
  for0(i, np) p[i].init(i);

  map<px, size_t> m; // setoid index
  size_t * lookup = (size_t *) alloc(sizeof(size_t) * np); // setoid lookup
  for0(i, np) lookup[i] = 0;
  size_t * s_i = (size_t *) alloc(sizeof(size_t) * np); // linear form of setoid index

  // printf("redundancy check..\n");
  size_t next_i = 0;
  for0(i, np){
    px * pi = &p[i];
    if(m.count(*pi) == 0){
      m[*pi] = i; // retain index to one element of setoid
      s_i[next_i] = i;
      lookup[i] = next_i ++;
    }
    else{
      cout << i << " redundant point: " << *pi << endl;
      lookup[i] = lookup[m[*pi]];
    }
  }

  if(m.size() < 1000){
    for0(i, np) printf("%zu ", lookup[i]);
    printf("\n");
  }

  size_t ms = m.size();
  printf("zero count %zu\n", nz);
  printf("redundant points %zu\n", np - ms);
  printf("setoid count %zu\n", ms);
  printf("data points total %zu\n", np);

  i = 0;
  size_t * idx = (size_t *) alloc(sizeof(size_t) * m.size());
  map<px, size_t>::iterator it;
  for(it = m.begin(); it != m.end(); it++){
    size_t L_i = it->second;
    // printf("%zu %zu %zu\n", i, L_i, s_i[i]);
    idx[i++] = s_i[i]; //L_i;
  }

  str ofn(bfn + str("_dedup")); // dedup file
  FILE * f = wopen(ofn);
  size_t n_r = fwrite(idx, sizeof(size_t), ms, f);
  fclose(f);
  printf("records written %zu\n", n_r);
  if(n_r != ms) err("unexpected record write count");

  str of2(bfn + str("_dedup_lookup")); // invert index?
  f = wopen(of2);
  n_r = fwrite(lookup, sizeof(size_t), np, f);
  fclose(f);
  printf("records written %zu\n", n_r);
  if(n_r != np) err("unexpected record write count");

  free(idx);
  m.clear();

  return 0;
}
