// 20210514 convert iq format data to PolSARPro complex format (one complex band envi type 6)
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4) err("Convert iq format data to PolSARPro binary format (envi type 6)\n convert_iq_to_cplx [i file] [q file] [output file]");
  str a(argv[1]);
  str b(argv[2]);

  str ah(hdr_fn(a));
  str bh(hdr_fn(b));

  size_t nr, nc, nb, np;
  hread(ah, nr, nc, nb);

  float * d_i = bread(a, nr, nc, nb); // read file
  float * d_q = bread(b, nr, nc, nb); // read file
  printf("nr %zu nc %zu nb %zu\n", nr, nc, nb);
  np = nr * nc;

  float * dd = falloc(np * 2); // allocate floats

  size_t j, j2;
  for0(j, np){
    j2 = 2 * j;
    dd[j2] = d_i[j];
    dd[j2 + 1] = d_q[j];
  }

  FILE * f = fopen(argv[3], "wb");
  if(!f) err("failed to open output file");

  size_t nw = fwrite(dd, sizeof(float), np * 2, f);
  if(nw != np * 2) err("unexpected write size");

  free(d_i);
  free(d_q);
  free(dd);

  str c(str(argv[3]) + str(".hdr"));
  hwrite(c, nr, nc, nb, 6);
  return 0;
}
