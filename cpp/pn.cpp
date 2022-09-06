/* 20220905 pn.cpp: simple "positive / negative" binary classifier */
#include"misc.h"

float * dat;
size_t nrow, ncol, nband, np;

float d(size_t i, size_t j){
  float dd = 0.;
  float di;
  size_t k, nk;
  for0(k, nband){
    nk = (np * nband);
    di = dat[i + nk] - dat[j + nk];
    dd += di * di;
  }
  return (float)sqrt((double)dd);
}

int main(int argc, char ** argv){

  str fn(argv[1]);
  if(argc < 2 || !exists(fn)) err("please check input file");

  str cfn(fn + str("_targets.csv"));

  vector<str> hdr;
  vector< vector<str>> lines(read_csv(cfn, hdr));

  if(hdr[0] != str("feature_id") || hdr[1] != str("row") || hdr[2] != str("lin")){
    err("check csv header");
  }

  // "row (= col) = j, line (= row) = i" format

  vector<long int> pi;
  vector<long int> pj;
  vector<long int> ni;
  vector<long int> nj;

  vector<vector<str>>::iterator it;
  for(it = lines.begin(); it != lines.end(); it++){
    vector<str> record(*it);
    if(record[0] == str("p")){
      pj.push_back(atol(record[1].c_str()));
      pi.push_back(atol(record[2].c_str()));
    }
    if(record[0] == str("n")){
      nj.push_back(atol(record[1].c_str()));
      ni.push_back(atol(record[2].c_str()));
    }
  }
  if(pi.size() == 0 || ni.size() == 0){
    err("both positive and negative sets must have at least one element");
  }

  cout << pi << endl;
  cout << pj << endl;
  cout << ni << endl;
  cout << nj << endl;

  str hfn(hdr_fn(fn));
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol;
  dat = bread(fn, nrow, ncol, nband);

  return 0;
}
