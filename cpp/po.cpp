/* 20220824 project onto */
#include"misc.h"

int main(int argc, char ** argv){
  str args("");
  int i;
  for0(i, argc - 1) args += (str(" ") + str(argv[i+1]));
  run(str("raster_project_onto.py") + args);
  return 0;
}
