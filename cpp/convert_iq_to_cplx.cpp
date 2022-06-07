/* 20210514 convert iq format data to PolSARPro complex format (one complex band envi type 6)
NB:
"in-phase and quadrature":
i-file and q-file are "real" and "imaginary" components of a complex number, respectively */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4){
    err("Convert iq format to PolSARPro complex format (envi type 6)\n convert_iq_to_cplx [i file] [q file] [output file]");
  }

  float * d_i, * d_q, * dd;
  size_t nr, nc, nb, np, j, j2;

  str a(argv[1]);
  str b(argv[2]);
  str ah(hdr_fn(a));
  str bh(hdr_fn(b));
  hread(ah, nr, nc, nb);

  (d_i = bread(a, nr, nc, nb)), (d_q = bread(b, nr, nc, nb));
  printf("nr %zu nc %zu nb %zu\n", nr, nc, nb);
  np = nr * nc;

  dd = falloc(np * 2); // allocate floats
  for0(j, np){
    j2 = 2 * j;
    dd[j2] = d_i[j];
    dd[j2 + 1] = d_q[j];
  }

  FILE * f = wopen(argv[3]);
  size_t nw = fwrite(dd, sizeof(float), np * 2, f);
  if(nw != np * 2) err("unexpected write size");
  str c(str(argv[3]) + str(".hdr"));
  hwrite(c, nr, nc, nb, 6);

  free(d_i);
  free(d_q);
  free(dd);
  return 0;
}
