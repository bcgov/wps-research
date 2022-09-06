/* 20220905 pn.cpp: simple "positive / negative" binary classifier */
#include"misc.h"
float * dat, * out;
size_t nrow, ncol, nband, np;

float d(size_t i, size_t j){
  // cout << "d " << i << " " << j << endl;
  float dd = 0.;
  float di;
  size_t k, nk;
  for0(k, nband){
    nk = (np * k);
    size_t ix = i + nk;
    size_t iy = j + nk;
    // cout << "ix " << ix << " iy " << iy << endl;
    di = dat[ix] - dat[iy];
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
  out = falloc(nrow * ncol); // allocate output buffer
  printf("dat[0] %f\n", dat[0]);

  size_t Np = pi.size(); // size of positive class
  size_t Nn = ni.size(); // size of negative class

  long int nni, npi;
  nni = npi = -1;
  float nnd, npd, dd;
  nnd = npd = FLT_MAX;

  size_t i, j, k;
  for0(i, nrow){
    for0(j, ncol){
      nni = npi = -1;
      nnd = npd = FLT_MAX;

      for0(k, Np){
	// cout << "k " << k << " pi[k] " << pi[k] << " pj[k] " << pj[k] << " nrow " << nrow << " ncol " << ncol << endl;
	size_t ix = i * ncol + j;
	size_t iy = pi[k] * ncol + pj[k];
	dd = d(ix, iy);
	if(k == 0){
          npd = dd;
	  npi = 0;
	}
	else{
	  if(dd < npd){
	    npd = dd;
	    npi = k;
	  }
	}
      }

      for0(k, Nn){
	// cout << "k " << k << endl;
        dd = d(i * ncol + j, ni[k] * ncol + nj[k]);
        if(k == 0){     
          nnd = dd;
          nni = 0;
        }
        else{
          if(dd < nnd){
            nnd = dd;
            nni = k;
          }
        }
      }  
      // should really be looking at the max, min, stdv of the distances to pos (neg) sets
      // printf("nnd %f nni %zu npd %f npi %zu\n", nnd, (size_t)nni, npd, (size_t)npi);
      out[i + ncol * j] = (float) (npd < nnd);
    }
  }

  str ofn(fn + str("_pn.bin"));
  str ohn(hdr_fn(ofn, true)); // out header file name

  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}
