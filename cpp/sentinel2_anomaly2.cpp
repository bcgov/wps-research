/* 20240625 sentinel2_anomaly.cpp "thermal" anomaly idea for sentinel2
   NB we didn't check if the headers of the two files match
*/
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 3){
    err("sentinel2_anomaly [pre date sentinel-2] [post date sentinel-2]");
  }

  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat1, *dat2, * b11, * b21, * b31, *b12, *b22, *b32; 
  long int bi[3];

  str fn1(argv[1]); /* binary files */
  str fn2(argv[2]);
  str hfn(hdr_fn(fn1)); /* headers */

  str ofn(fn1 + "_" + fn2 + "_ratio.bin");
  str ohn(hdr_fn(ofn, true));

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

  printf("reading pre-image %s\n", fn1.c_str());
  dat1 = bread(fn1, nrow, ncol, nband); /* read the data */
  b11 = &dat1[bi[0]]; /* select the bands */
  b21 = &dat1[bi[1]];
  b31 = &dat1[bi[2]];

  printf("reading post-image %s\n", fn2.c_str());
  dat2 = bread(fn2, nrow, ncol, nband); /* read the data */

  b12 = &dat2[bi[0]]; /* select the bands */
  b22 = &dat2[bi[1]];
  b32 = &dat2[bi[2]];

  printf("allocating output buffer..\n");
  out = falloc(np * 3);

  printf("processing data..\n");
  for0(i, np){
    out[i]   = (b12[i] - b11[i]) / (b12[i] + b11[i]);
    out[i + np]      = (b22[i] - b21[i]) / (b22[i] + b21[i]);
    out[i + np + np]           = (b32[i] - b31[i]) / (b32[i] + b31[i]);
  }

  printf("writing output..\n");
  vector<str> bn;
  bn.push_back(date_s + str("(b32[i] - b31[i]) / (b32[i] + b31[i]) sentinel2_anomaly.cpp"));
  bn.push_back(date_s + str("(b22[i] - b21[i]) / (b22[i] + b21[i]) sentinel2_anomaly.cpp"));
  bn.push_back(date_s + str("(b12[i] - b11[i]) / (b12[i] + b11[i]) sentinel2_anomaly.cpp"));
 
  hwrite(ohn, nrow, ncol, 3, 4, bn);
  printf("+w %s\n", ofn.c_str());
  bwrite(out, ofn, nrow, ncol, 3);
  run(str("envi_header_copy_mapinfo.py ") + hfn + str(" ") + ohn);  
  free(dat1);
  free(dat2);
  free(out);
  return 0;
}
