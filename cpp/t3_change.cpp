#include"matrix3.h"
#include"misc.h"

int main(int argc, char ** argv){
  str sep("/");
  str channels[9] = {str("T11.bin"),
		     str("T12_imag.bin"),
		     str("T12_real.bin"),
		     str("T13_imag.bin"),
		     str("T13_real.bin"),
		     str("T22.bin"),
		     str("T23_imag.bin"),
		     str("T23_real.bin"),
		     str("T33.bin")};

  if(argc < 3){
    err("t3_change.py [t3 path 1] [t3 path 2]");
  }

  size_t nr, nc, nb, np, i, j;
  float * a[9];
  float * b[9];
  
  for0(i, 9){
    a[i] = NULL;
    str fn(str(argv[1]) + sep + channels[i]);
    str hfn(hdr_fn(str(argv[1]) + sep + channels[i]));
    hread(hfn, nr, nc, nb);
    a[i] = bread(fn, nr, nc, nb);   
  }

  for0(i, 9){
    b[i] = NULL;
    str fn(str(argv[2]) + sep + channels[i]);
    str hfn(hdr_fn(str(argv[2]) + sep + channels[i]));
    hread(hfn, nr, nc, nb);
    b[i] = bread(fn, nr, nc, nb);
  }

  np = nr * nc;
  for0(i, np){
    herm3<cf> A;
    herm3<cf> B;
     /*void initT3(float T11R, float T12I, float T12R, float T13I, float T13R,
              float T22R, float T23I, float T23R, float T33R) */

    A.initT3(a[0][i], a[1][i], a[2][i], a[3][i], a[4][i], a[5][i], a[6][i], a[7][i], a[8][i]);
    B.initT3(b[0][i], b[1][i], b[2][i], b[3][i], b[4][i], b[5][i], b[6][i], b[7][i], b[8][i]);

  }



  return 0;
}
