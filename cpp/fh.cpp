/* fix envi header */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("fh [envi header file name .hdr] # fix header file");

  run(str("envi_header_cleanup.py ") + str(argv[1]));
  return 0;
}
