/* 20200128 count occurrence of float variable */
#include<stdio.h>
#include<stdlib.h>
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("class_count.exe [input file name]");

  map<float, size_t> count;
  FILE * f = ropen(argv[1]);
  size_t fs = fsize(argv[1]);
  size_t n_nan, nf = fs / sizeof(float);
  float di, * d = (float *) alloc(fs);
  fread(d, sizeof(float), nf, f);

  n_nan = 0;
  for(size_t i = 0; i < nf; i++){
    di = d[i];
    if(isnan(di) || isinf(di)){
      n_nan += 1;
    }
    else{
      if(count.count(di) < 1) count[di] = 0;
      count[di] += 1;
    }
  }
  fclose(f); // count[NAN] = n_nan; // the map can't assign to NAN

  if(count.size() == 0 && n_nan ==0) return 0;
  cout << "{";

  map<float, size_t>::iterator it = count.begin();
  cout << it->first << ":" << it->second;

  while(++it != count.end()){
    cout << "," << endl << it->first << ":" << it->second;
  }

  if(n_nan > 0) cout << "," << endl << "NAN:" << n_nan;
  cout << "}" << endl;


  cout << "number of distinct values:" << count.size() << endl;
  free(d);
  return 0;
}
