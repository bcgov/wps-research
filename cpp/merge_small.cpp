/* 20240803 
*/
#include "misc.h"

int main(int argc, char ** argv){
  str small("small");
  if(!exists(small)){
    run(str("mkdir small"));
  }
  run(str("raster_warp_all.py -s 4 ./ ./small"));
  run(str("cd small; merge2.py"));
  return 0;
}


