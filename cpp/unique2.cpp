/* 20230424 unique2.cpp: deduplicate records by line. 20240226: no dependencies
Input file needs to fit in RAM

compilation:
  g++ unique2.cpp -O3 -o unique2

running: e.g.:
  unique2 EarthNetworks_BCWS_LX_2023.csv 

  wc -l EarthNetworks_BCWS_LX_2023.csv
  wc -l EarthNetworks_BCWS_LX_2023.csv_unique.csv

*/
#include<unordered_set>
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<string>
#define str string
using namespace std;

int err(const char * m){
  cout << "Error: " << m << endl; 
  exit(1);
}

int main(int argc, char ** argv){
  if(argc < 2) err("unique2.cpp [input text file name]\n");

  str fn(argv[1]);
  str ofn(fn + str("_unique.txt"));
  
  ifstream dfile(fn);
  if(!dfile.is_open()) err((str("failed to open file:") + fn).c_str());
  
  str line;
  unordered_set<str> d;

  while(getline(dfile, line)) d.insert(line);
  dfile.close();

  ofstream outfile(ofn);
  if(!outfile.is_open()) err((str("failed to open file:") + ofn).c_str());

  unordered_set<str>::iterator it;
  for(it = d.begin(); it != d.end(); it++) outfile << *it << endl;
  outfile.close();

  return 0;
}

