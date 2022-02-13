#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("sentinel2_active.exe [input file name]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));

  size_t nrow, ncol, nband;
  int bi[3];
  int j;
  bi[0] = bi[1] = bi[2] = -1;
  vector<string> s;
  vector<string> t;
  t.push_back(str("945n"));
  t.push_back(str("1610n"));
  t.push_back(str("2190n"));
  hread(hfn, nrow, ncol, nband, s);

  int i, n;
  n = s.size();
  for0(i, n){
    for0(j, 3){
      if(contains(s[i], t[j])){
        bi[j] = i;
      }
    }
  }

  for0(j, 3){
    cout << "\t" << bi[j] << endl;
  }

  return 0;
}
