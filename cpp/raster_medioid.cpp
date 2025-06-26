/* 20250626 compute medioid for a list of rasters. Use random file access, to avoid loading the whole files in memory */

#include"misc.h"
int main(int argc, char ** argv){

  int i;
  for0(i, argc){
    printf("arg %d [%s]\n", i, argv[i]);
  }

  return 0;
}


