/* 20240803 
*/
#include "misc.h"

int main(int argc, char ** argv){
  str small("small");
  if(!exists(small)){
    run(str("mkdir small"));
  }
 
  // clean up bad stuff 
  run("find ./ -name \"merge*.bin\" | xargs rm");
  run("find ./ -name \"tmp*.bin\" | xargs rm");
  run("find ./ -name \"*.vrt\" | xargs rm");
  run("clean");

  // resample the imagery to smaller
  run(str("raster_warp_all.py -s 4 ./ ./small"));

  // run the merging (could this be better parallelised)
  run(str("cd small; merge2.py"));
  return 0;
}


