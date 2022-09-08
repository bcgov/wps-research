/* 20220906: convert PNG file to bin file */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("png2bin [input .png filename]");
  }
  str fn(argv[1]);
  str ofn(fn + str(".bin"));
   
  run(str("gdal_translate -of ENVI -ot Float32 -b 1 ") + fn + str(" ") + ofn);

  return 0;
}
