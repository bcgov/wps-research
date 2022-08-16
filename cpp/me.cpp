/* 20220815 make directory and enter it */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("me [directory name] # make directory and enter it");

  str dn(argv[1]);
  system(str("mkdir ") + dn);
  system(str("cd ") + dn);
  return 0;
}

