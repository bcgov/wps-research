/* 20220822 "fix header". Ways to run:

   fh envi_file.hdr # full filename specified
   fh envi_file     # .hdr extension assumed
   fh   	    # run on all header files in present dir */
#include"misc.h"

vector<str> good;
void runme(size_t i){
  run(str("envi_header_cleanup.py ") + good[i]);
}

int main(int argc, char ** argv){
  if(argc < 2){
    vector<str> lines(split(exec("ls -1 *.hdr"), '\n'));
    for(vector<str>::iterator it = lines.begin(); it != lines.end(); it++){
      if(exists(*it)){
        good.push_back(*it);
      }
    }
    cout << lines << endl;
    if(good.size() < 1){
      err("fh [envi header file name .hdr] # fix header file");
    }
  }
  else{
    str fn(argv[1]);
    if(exists(fn)){
      good.push_back(fn);
    }
    else{
      if(exists(fn + str(".hdr"))){
        fn += str(".hdr");
	good.push_back(fn);
      }
    }
  }
  parfor(0, good.size(), runme);
  return 0;
}
