/* 20220818 htrim .5 then dominant, extract band 1 */
#include "misc.h"

int main(int argc, char ** argv){

  str f("sub.bin");
  str swir(f + str("_swir.bin"));
  if(!exists(swir)){
    run(str("sentinel2_swir_subselect sub.bin"));
  }

  str ht(swir + str("_ht.bin"));
  str dom(ht + str("_dominant.bin"));

  run(str("htrim2 ") + swir + str(" ") + (argc < 2 ? str(".5 .5") : (str(argv[1]) + str(" ") + str(argv[1]))));

  if(!exists(ht)){
	  printf("Error: could not find file: %s\n", ht.c_str());
	  err("file not found");
  }

  run(str("raster_dominant.exe " + ht));
  run(str("unstack.exe ") + dom + str(" 1"));

  run(str("python3 ~/GitHub/wps-research/py/envi_header_copy_mapinfo.py sub.hdr sub.bin_swir.bin_ht.bin_dominant.bin_001.hdr"));
  run(str("clean"));
  run(str("envi2tif.py sub.bin_swir.bin_ht.bin_dominant.bin_001.bin"));
  run(str("imv sub.bin_swir.bin_ht.bin_dominant.bin_001.bin"));
  return 0;
}


