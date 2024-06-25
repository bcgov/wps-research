/* 20240625 sentinel2_anomaly.cpp "thermal" anomaly idea for sentinel2
   NB we didn't check if the headers of the two files match
*/
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 3){
    err("sentinel2_anomaly [pre date sentinel-2] [post date sentinel-2]");
  }

  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat1, *dat2, * b11, * b21, * b31, *b12, *b22, *b32, *out2, *out3;
  long int bi[3];
 

  str fn1(argv[1]); /* binary files */
  str fn2(argv[2]);
  str ofn(fn2 + "_anomaly.bin");
  str hfn(hdr_fn(fn1)); /* headers */
  str hf2(hdr_fn(ofn, true));

  str ofn2(fn1 + "_ratio.bin");
  str ofn3(fn2 + "_ratio.bin");
  str hf3(hdr_fn(ofn2, true));
  str hf4(hdr_fn(ofn3, true));

  vector<string> s, t; /* band names + ones to use */
  t.push_back(str("2190nm"));
  t.push_back(str("1610nm"));
  t.push_back(str("945nm"));

  hread(hfn, nrow, ncol, nband, s);
  for0(i, 3) bi[i] = -1;
  np = nrow * ncol;
  n = s.size(); 
  
  str date_s;
  for0(i, n){
    for0(j, 3){
      if(contains(s[i], t[j])){
        bi[j] = i * np; /* found a required band */
        printf("bi[%zu]=%zu \"%s\"\n", j, bi[j], s[i].c_str());
	      vector<string> w(split(s[i], ' '));
	      date_s = w[0]; /* assume datetime string at beginning */
      }
    }
  }
  for0(i, 3) if(bi[i] < 0){
    printf("Missing band: %s\n", t[i].c_str());
    err("Missing band");
  }

  dat1 = bread(fn1, nrow, ncol, nband); /* read the data */
  b11 = &dat1[bi[0]]; /* select the bands */
  b21 = &dat1[bi[1]];
  b31 = &dat1[bi[2]];

  dat2 = bread(fn2, nrow, ncol, nband); /* read the data */

  b12 = &dat2[bi[0]]; /* select the bands */
  b22 = &dat2[bi[1]];
  b32 = &dat2[bi[2]];

  out = falloc(np);
  out2 = falloc(np);
  out3 = falloc(np);
  for0(i, np){
    float ratio1 = (b11[i] - b21[i]) / (b11[i] + b21[i]);
    float ratio2 = (b12[i] - b22[i]) / (b12[i] + b22[i]);
    out2[i] = ratio1;
    out3[i] = ratio2;
    out[i] = (ratio1 - ratio2) / (ratio2 + ratio1);
  }

  vector<str> bn;
  bn.push_back(date_s + str(" sentinel2_anomaly.cpp"));
  hwrite(hf2, nrow, ncol, 1, 4, bn);
  bwrite(out, ofn, nrow, ncol, 1);

  bn.clear();
  bn.push_back(date_s + str(" sentinel2_anomaly.cpp"));
  hwrite(hf3, nrow, ncol, 1, 4, bn);
  bwrite(out2, ofn2, nrow, ncol, 1);

  bn.clear(); 
  bn.push_back(date_s + str(" sentinel2_anomaly.cpp"));
  hwrite(hf4, nrow, ncol, 1, 4, bn);
  bwrite(out3, ofn3, nrow, ncol, 1);

  run(str("envi_header_copy_mapinfo.py ") + hfn + str(" ") + hf2);
  run(str("envi_header_copy_mapinfo.py ") + hfn + str(" ") + hf3);
  run(str("envi_header_copy_mapinfo.py ") + hfn + str(" ") + hf4);

  free(dat1);
  free(dat2);
  free(out);

  return 0;
}
