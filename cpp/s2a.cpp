/* 20220811 */
#include "misc.h"

int main(int argc, char ** argv){
  str f("sub.bin");
  str swir(f + str("_swir.bin"));
  str s_d(swir + str("_dominant.bin"));
  
  run(str("sentinel2_active.exe " + f));
  run(str("sentinel2_swir_subselect.exe " + f));
  run(str("raster_dominant.exe " + swir));
  run(str("unstack.exe ") + s_d + str(" 1"));

  return 0;
}


