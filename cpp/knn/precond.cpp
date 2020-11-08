/* preconditioning for patch data. Null pixels within a patch are assigned the average of non-null pixels within a patch, if non-zero */
#include"../misc.h"

float * dat;
size_t nr, nc, nb, nf, np, ps, fpp; // image dimensions

class px{
public:
  float * d;
  void init(size_t ix){
    d = &dat[ix * fpp];
  }

};

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
