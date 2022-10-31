/* 20221030 raster_summarize.cpp: count vector data 


Next step motivated: class transition matrix (succesion)! "probability"
*/
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_summarize [raster cube] # count vector data: suggest not for large number of obsevations\n");
  size_t nrow, ncol, nband, np, nf;
  float * out, * dat, d;
  size_t i, j, k;

  str fn(argv[1]);
  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn));
  vector<str> s;
  hread(hfn, nrow, ncol, nband, s); // read header

  cout << s << endl;
  np = nrow * ncol; // number of input pix
  nf = np * nband;  

  double total = 0.;
  dat = bread(fn, nrow, ncol, nband);
  vector<float> v;
  map< vector<float>, size_t> count;
  for0(i, np){
    v.clear();

    for0(k, nband){
	    d = dat[i + (k * np)];
      v.push_back(d);
    }
    if(count.count(v) < 1){
      count[v] = 1;
    }
    else{
      count[v] += 1;
    }
  }

  map< vector<float>, size_t>::iterator it;
  for(it = count.begin(); it != count.end(); it ++){
   //cout << it->first << " " << it->second << endl;
   total += (double) it->second; 
  }
  int ci = 0;
  for(it = count.begin(); it != count.end(); it ++){
    cout << it->first << " " << it->second;
    cout << " " << ((double)(it->second) / total);
    cout << " %" << 100. * ((double)(it->second) / total);
    cout << " " << s[ci ++] << endl;
  }
  printf("grand total: %e\n", total);
  free(dat); 
  return 0;
}
