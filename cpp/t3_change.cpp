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

  int i;
  size_t nr, nc, nb, np;
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
  }



  return 0;
}
