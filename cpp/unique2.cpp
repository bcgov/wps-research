/* 20230424 unique2.cpp: deduplicate records by line */
#include "misc.h"
int main(int argc, char ** argv){
  if(argc < 2)
    err("unique2.cpp [input text file name]\n");
  str fn(argv[1]);
  str ofn(fn + str("_unique.txt"));
  
  ifstream dfile(dfn);
  if(!dfile.is_open())
    err(string("failed to open input data file:") + dfn);

  unordered_set<str> d;
  str line;

  while(getline(dfile, line)){
    d.insert(line);
  }
  dfile.close();

  ofstream outfile(ofn);
  if(!outfile.is_open())
    err(string("failed to write-open file:") + ofn);

  unordered_set<str>::iterator it;
  for(it = d.begin(); it != d.end(); it++)
    outfile << *it << endl;

  outfile.close();

  return 0;
}

