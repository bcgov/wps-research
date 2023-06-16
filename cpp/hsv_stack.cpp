/* 20220811 */
#include "misc.h"

int main(int argc, char ** argv){
  str f("sub.bin");
  run(str("htrim2 ") + f + str(" .25"));
  run(str("rgb2hsv sub.bin_ht.bin"));
  run("raster_stack.py sub.bin_ht.bin sub.bin_ht.bin_hsv.bin stack.bin");
  run("envi_header_copy_mapinfo.py sub.hdr stack.hdr");   
  run("~/GitHub/kgc2010/kgc.exe stack.bin 33333 700 0");

  run("cd output");
  run("cp ../sub.bin_ht.bin_smult.tif ./");
  run("envi_header_copy_mapinfo.py ../sub.hdr ./out_binary.hdr");
  run("envi2tif.py out_binary.bin");

  run("cp ../sub.bin .");
  run("cp ../sub.hdr .");
  run("envi2tif.py sub.bin");


  run("gimp *.tif");
  
  return 0;
}
