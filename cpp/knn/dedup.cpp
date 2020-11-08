/* preconditioning for patch data. Null pixels within a patch are assigned the average of non-null pixels within a patch, if non-zero */
#include"../misc.h"

float * dat;
size_t nr, nc, nb, nf, np, ps, fpp; // image dimensions

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
      if(isinf(-x) || isnan(-x) || isinf(d) || isnan(d)){
        bad = true;
        break;
      }
      if(bad){
	      printf("warning: bad data ix %zu\n", ix * fpp);
        for0(i, fpp) d[i] = 0.;
      }
    }
  }
};

// output operator for set container
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
  if(argc < 2) err("dedup [input fp stack .bin name, already patched]");

  size_t i, j, k;
  str bfn(argv[1]);
  str hfn(hdr_fn(bfn));
  hread(hfn, nr, nc, nb);
  printf("nr %zu nc %zu nb %zu\n", nr, nc, nb);

  str pfn(bfn + str("_patch"));
  nf = fsize(pfn) / sizeof(float); //number of floats

  ps = int_read(bfn + str("_ps")); // patch size
  nb = int_read(bfn + str("_nb")); // number of actual image data bands (not incl. groundref)
  fpp = ps * ps * nb; // number of floats per patch
  np = nf / fpp; // number of patches

  printf("ps %zu\n", ps); // patch size
  printf("fpp %zu\n", fpp);
  printf("np: %zu\n", np);

  dat = float_read(pfn, nf); // read patch data, as linear array of floats
  if(nf != np * fpp) err("unexpected number of floats read");

  px * p = new px[np];
  for0(i, np) p[i].init(i);

  return 0;
}
