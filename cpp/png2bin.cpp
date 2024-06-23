/* 20220906: convert PNG file to bin file */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("png2bin [input .png filename] [optional factor: do not scale result to [0,1]]");
  }
  str fn(argv[1]);
  str ofn(fn + str(".bin"));
  str ohn(fn + str(".hdr")); 

  run(str("gdal_translate -of ENVI -ot Float32 -b 1 ") + fn + str(" ") + ofn);
  run(str("fh ") + ohn);

  if(argc < 3){
    run(str("raster_scale ") + ofn);
    run(str("cp ") + ofn + str("_scale.bin ") + ofn);
  }

  if(exists(str("result.bin_thres.hdr"))){
   // run(str("cp result.bin_thres.hdr " + ohn));
  }
  else{
    run(str("envi_header_copy_mapinfo.py sub.hdr result.png.hdr"));
  }

  return 0;
}
