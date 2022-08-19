/* 20220811 */
#include "misc.h"

int main(int argc, char ** argv){
  str f("sub.bin");
  str swir(f + str("_swir.bin"));
  str s_d(swir + str("_dominant.bin"));
  str s_d1(s_d + str("_001.bin"));
  str a(f + str("_active.bin"));

  run(str("sentinel2_active.exe " + f));
  run(str("sentinel2_swir_subselect.exe " + f));
  run(str("raster_dominant.exe " + swir));
  run(str("unstack.exe ") + s_d + str(" 1"));

  run(str("raster_sum.exe ") + a + str(" ") + s_d1 + str(" result.bin"));

  run(str("raster_threshold.exe result.bin GEQ 1."));
  run(str("python3 ~/GitHub/wps-research/py/envi_header_copy_mapinfo.py sub.hdr result.bin_thres.hdr"));

  run(str("python3 ~/GitHub/wps-research/py/binary_polygonize.py result.bin_thres.bin"));
  run(str("python3 ~/GitHub/wps-research/py/envi2tif.py ") + swir);
  run(str("clean"));
  cout << str("imv result.bin_thres.bin") << endl;
  return 0;
}


