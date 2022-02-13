#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("sentinel2_active.exe [input file name]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));

  size_t nrow, ncol, nband, np, i, j, k, n;
  int bi[3];
  bi[0] = bi[1] = bi[2] = -1;
  vector<string> s;
  vector<string> t;
  t.push_back(str("945n"));
  t.push_back(str("1610n"));
  t.push_back(str("2190n"));
  hread(hfn, nrow, ncol, nband, s);
  np = nrow * ncol;
  n = s.size();
  for0(i, n) for0(j, 3) if(contains(s[i], t[j])) bi[j] = i;
  for0(j, 3) cout << "\t" << bi[j] << endl;
  float * dat = bread(fn, nrow, ncol, nband);

  int n_out = 1; // 6;
  float * out = falloc(np * n_out);
  
  size_t ij;
  size_t bi0, bi1, bi2;
  bi0 = np * bi[0]; 
  bi1 = np * bi[1];
  bi2 = np * bi[2];

  size_t np0, np1, np2, np3, np4, np5;
  np0 = 0;
  np1 = np;
  np2 = np * 2;
  np3 = np * 3;
  np4 = np * 4;
  np5 = np * 5;
  for0(i, nrow) for0(j, ncol){
	  ij = i * ncol + j;
	  out[ij + np0] = dat[ij + bi2] - dat[ij + bi1] > 175.;
	  // out[ij + np1] = dat[ij + bi2] - dat[ij + bi1];
	  // out[ij + np2] = out[ij + bi2] - out[ij + bi0];
	  // out[ij + np3] = out[ij + np2] - out[ij + np1];
	  //out[ij + np4] = out[ij + np1] > out[ij + np0];
	  // out[ij + np4] = out[ij + np1] > 4.*out[ij + np3]; 
	  //out[ij + np5] = out[ij + np2] > (.5 * out[ij + np4] + .5 * out[ij + np1]);
  }


  str ofn(fn + "_active.bin");
  bwrite(out, ofn, nrow, ncol, n_out);
  str hf2(hdr_fn(ofn, true));
  hwrite(hf2, nrow, ncol, n_out);

  //  //free(dat);
  //free(out);
  return 0;
}
