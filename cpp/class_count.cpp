#include<stdio.h>
#include<stdlib.h>
#include"misc.h"

int main(int argc, char ** argv){

  if(argc < 2) err("class_count.exe [input file name]");

  FILE * f = ropen(argv[1]);
  
  size_t fs = fsize(argv[1]);
  size_t nf = fs / sizeof(float);
  float * d = (float *) alloc(fs);
  fread(d, sizeof(float), nf, f);
  map<float, size_t> count;

  float di;
  size_t  n_nan = 0;
  for(size_t i = 0; i < nf; i++){
    di = d[i];
    if(isnan(di) || isinf(di)){
     n_nan += 1;
     continue;
    }
    if(count.count(di) < 1){
      count[di] = 0;
    }
    count[di] += 1;
  }
  fclose(f);
   
  cout << count << endl;
  cout << "nan: " << n_nan << endl;
  cout << "total: " << nf << endl;
  cout << "non_nan: " << nf - n_nan << endl;

  return 0;

}

