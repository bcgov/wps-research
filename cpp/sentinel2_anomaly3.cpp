/* 20260122: modified to process all available bands.  
   20240625 sentinel2_anomaly.cpp "thermal" anomaly idea for sentinel2
   20250122 modified to process all bands, matching by portion after last colon
   NB we didn't check if the headers of the two files match
*/

/* 20240625 sentinel2_anomaly.cpp "thermal" anomaly idea for sentinel2
   20250122 modified to process all bands, matching by portion after last colon
   NB we didn't check if the headers of the two files match
*/
#include"misc.h"
#include <filesystem>

str after_last_colon(const str& s) {
  size_t pos = s.rfind(':');
  if (pos == string::npos) return s;
  return s.substr(pos + 1);
}

int main(int argc, char ** argv){
  if(argc < 3){
    err("sentinel2_anomaly [pre date sentinel-2] [post date sentinel-2] [--divide]");
  }

  bool divide_mode = false;
  for(int a = 3; a < argc; a++){
    if(str(argv[a]) == str("--divide")){
      divide_mode = true;
    }
  }

  size_t nrow, ncol, nband, np, i, j;
  size_t nrow2, ncol2, nband2;
  float * out, * dat1, *dat2;
  str fn1(argv[1]); /* binary files */
  str fn2(argv[2]);

  /* discard the absolute path to create the output filename e.g. so we can run from ramdisk */
  std::filesystem::path fn_1 = fn1;
  std::filesystem::path fn_2 = fn2;
  str hfn1(hdr_fn(fn_1)); /* headers */
  str hfn2(hdr_fn(fn_2));

  str ofn(divide_mode ? "ratio_divide.bin" : "ratio.bin");
  str ohn(hdr_fn(ofn, true));
  str formula(divide_mode ? " post/pre" : " (post-pre)/(post+pre)");

  cout << "output filename: " << ofn << endl;
  cout << "output header:   " << ohn << endl;
  cout << "mode: " << (divide_mode ? "divide" : "normalized difference") << endl;

  vector<string> s1, s2; /* band names from each file */
  hread(hfn1, nrow, ncol, nband, s1);
  hread(hfn2, nrow2, ncol2, nband2, s2);

  if(nrow != nrow2 || ncol != ncol2){
    err("Image dimensions do not match");
  }
  if(nband != nband2){
    err("Number of bands do not match");
  }

  np = nrow * ncol;

  /* check that band name suffixes match */
  for(i = 0; i < nband; i++){
    str suffix1 = after_last_colon(s1[i]);
    str suffix2 = after_last_colon(s2[i]);
    if(suffix1 != suffix2){
      printf("Band %zu mismatch: '%s' vs '%s'\n", i, suffix1.c_str(), suffix2.c_str());
      err("Band name suffix mismatch");
    }
    printf("Band %zu: '%s'\n", i, suffix1.c_str());
  }

  printf("reading pre-image %s\n", fn1.c_str());
  dat1 = bread(fn1, nrow, ncol, nband);
  printf("reading post-image %s\n", fn2.c_str());
  dat2 = bread(fn2, nrow, ncol, nband);

  printf("allocating output buffer..\n");
  out = falloc(np * nband);

  printf("processing data..\n");
  for(i = 0; i < nband; i++){
    float * pre = &dat1[i * np];
    float * post = &dat2[i * np];
    float * o = &out[i * np];
    if(divide_mode){
      for(j = 0; j < np; j++){
        o[j] = post[j] / pre[j];
      }
    } else {
      for(j = 0; j < np; j++){
        o[j] = (post[j] - pre[j]) / (post[j] + pre[j]);
      }
    }
  }

  printf("writing output..\n");
  vector<str> bn;
  for(i = 0; i < nband; i++){
    bn.push_back(str("anomaly:") + after_last_colon(s1[i]) + formula);
  }

  hwrite(ohn, nrow, ncol, nband, 4, bn);
  printf("+w %s\n", ofn.c_str());
  bwrite(out, ofn, nrow, ncol, nband);
  run(str("envi_header_copy_mapinfo.py ") + hfn1 + str(" ") + ohn);
  free(dat1);
  free(dat2);
  free(out);
  return 0;
}
