#include"misc.h"
int main(int argc, char **argv){
  if(argc < 2) err("kgc.exe [input binary file bsq]");
  str fn(argv[1]);
  run(str("bsq2bip.exe ") + fn);
  run(str("precond.exe ") + fn + str("_bip.bin"));
  run(str("bip_dedup.exe ") + fn + str("_bip.bin_precond.bin"));
  run(str("kgc2020.exe ") + fn + str("_bip.bin_precond.bin ") + fn + str("_bip.bin_precond.bin_dedup"));
  return 0;
}
