#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("sentinel2_active.exe [input file name]");
  }

  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat, * b1, * b2, * b3;
  long int bi[3];

  str fn(argv[1]); /* binary files */
  str ofn(fn + "_active.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));

  vector<string> s, t; /* band names + ones to use */
  t.push_back(str("945nm"));
  t.push_back(str("1610nm"));
  t.push_back(str("2190nm"));

  hread(hfn, nrow, ncol, nband, s);
  for0(i, 3) bi[i] = -1;
  np = nrow * ncol;
  n = s.size();

  for0(i, n){
    for0(j, 3){
      if(contains(s[i], t[j])){
        bi[j] = i * np; /* found a required band */
        printf("bi[%zu]=%zu \"%s\"\n", j, bi[j], s[i].c_str());
      }
    }
  }
  for0(i, 3) if(bi[i] < 0){
    printf("Missing band: %s\n", t[i].c_str());
    err("Missing band");
  }

  dat = bread(fn, nrow, ncol, nband); /* read the data */
  b1 = &dat[bi[0]]; /* select the bands */
  b2 = &dat[bi[1]];
  b3 = &dat[bi[2]];
  out = falloc(np);

  for0(i, np){
    out[i] = (float)(b2[i] - b1[i]) > 175.;
    out[i] *= (float)(b3[i] > b2[i]); /* reduce fp? */
  }

  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(hf2, nrow, ncol, 1);
  free(dat); /* plot spectra? */
  free(out);
  return 0;
}