/* fix envi header */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("fh [envi header file name .hdr] # fix header file");

  str fn(argv[1]);
  if(exists(fn)){
  }
  else{
    if(exists(fn + str(".hdr"))){
      fn += str(".hdr");
    }
  }

  run(str("envi_header_cleanup.py ") + fn);
  return 0;
}
