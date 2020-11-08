/* preconditioning for patch data. Null pixels within a patch are assigned the average of non-null pixels within a patch, if non-zero */
#include"../misc.h"

size_t nr, nc, nb, np; // image dimensions

int main(int argc, char ** argv){
  if(argc < 2) err("dedup [input fp stack .bin name, already patched]");

  str bfn(argv[1]);
  str hfn(hdr_fn(bfn));
  hread(hfn, nr, nc, nb);
  printf("nr %zu nc %zu nb %zu\n", nr, nc, nb);

  str pfn(bfn + str("_patch"));
  np = fsize(pfn) / sizeof(float); //number of patches

  size_t ps, nb, fpp;
  ps = restore_int(bfn + str("_ps"));  
  nb = restore_int(bfn + str("_nb"));
  fpp = ps * ps * nb;

  printf("ps %zu\n", ps);
  printf("fpp %zu\n", fpp);
  printf("nfloats: %zu\n", np);
  printf("npatch:  %zu\n", np / fpp); 

}

