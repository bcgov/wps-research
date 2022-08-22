/* fix envi header */
#include"misc.h"
int main(int argc, char ** argv){
  vector<str> good;
  
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

  for(vector<str>::iterator it = good.begin(); it != good.end(); it++){
    run(str("envi_header_cleanup.py ") + *it);
  }
  return 0;
}
