#include"misc.h"
int main(int argc, char **argv){
  if(argc < 2) err("kgc.exe [input binary file bsq] [--nskip skip_factor]");
  str fn(argv[1]);
  
  // parse --nskip argument
  str nskip_arg("");
  for(int a = 2; a < argc; a++){
    if(strcmp(argv[a], "--nskip") == 0 && a + 1 < argc){
      nskip_arg = str(" --nskip ") + str(argv[a + 1]);
      a++;
    }
  }
  
  run(str("bsq2bip.exe ") + fn);
  run(str("precond.exe ") + fn + str("_bip.bin"));
  run(str("bip_dedup.exe ") + fn + str("_bip.bin_precond.bin"));
  run(str("kgc2020 ") + fn + str("_bip.bin_precond.bin ") + fn + str("_bip.bin_precond.bin_dedup") + nskip_arg);
  return 0;
}
