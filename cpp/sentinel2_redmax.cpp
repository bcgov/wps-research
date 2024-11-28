/* 20241128 sentinel2_redmax.cpp find the "reddest" pixels in a sentinel-2 sequence

Caveat:  Assume that the longest wavelength ( B12 ) is first! */
#include"misc.h"

int main(int argc, char ** argv){
  int a = system("rm sentinel2_redmax.bin tmp*bin");

  vector<str> lines( split(exec("ls -1 *.bin"), '\n'));

  size_t i, nrow, ncol, nband, np, k, nrow2, ncol2, nband2;

  for(vector<str>::iterator it = lines.begin(); it != lines.end(); it++){
    i = 0;
    str fn(*it);
    str hfn(hdr_fn(fn));  // input header file name

    if(i==0){
      hread(hfn, nrow, ncol, nband);
    }
    else{
      hread(hfn, nrow2, ncol2, nband2);
      if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
        err(str("file: ") + fn + str(" has different dimenions than ") + lines[0]);
      }
    }
    i ++;
  }
  np = nrow * ncol;
  // now that we are sure the files match, proceed!

  float * out = falloc(np * nband);

  i = 0 ;
  for(vector<str>::iterator it = lines.begin(); it != lines.end(); it++){
    str fn(*it);
    str hfn(hdr_fn(fn));  // input header file name

    cout << "+r" << *it << endl;
    float * dat = bread(*it, nrow, ncol, nband);

    if(i==0){
      for0(k, np * nband) out[k] = dat[k];
    }
    else{
      // for each pixel, find out if the updated version is "more red"

      for0(i, np){
        float red1 = 0.;
        float red2 = 0.;
        bool new_good = true;
        bool old_good = true;
        for0(k, nband){
          if(isnan(dat[np*k + i])){
              new_good = false;
          }
          if(isnan(out[np*k + i])){
            old_good = false;
          }
          red1 += out[np * k + i]; // add up the bands, this pixel ( candidate result so far ) 
          red2 += dat[np * k + i]; // current image being processed, this pixel 
        }

        if(!new_good) continue;

        red1 = out[i] / red1;  // red as fraction of sum of bands
        red2 = dat[i] / red2;  // red as fraction of sum of bands
        if(red2 > red1 || (!old_good)){
          for0(k, nband) out[np * k + i] = dat[np * k + i];
        }
      }
    }

    i++;   
  }

  bwrite(out, str("sentinel2_redmax.bin"), nrow, ncol, nband);
  run((str("cp -v ") + hdr_fn(lines[0]) + str(" sentinel2_redmax.hdr")).c_str());

 

/*
size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat1, *dat2, * b11, * b21, * b31, *b12, *b22, *b32; 
  long int bi[3];

  str fn1(argv[1]); 
  str fn2(argv[2]);
  str hfn(hdr_fn(fn1));

  str ofn(fn1 + "_" + fn2 + "_ratio.bin");
  str ohn(hdr_fn(ofn, true));

  vector<string> s, t;
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
        bi[j] = i * np;
        printf("bi[%zu]=%zu \"%s\"\n", j, bi[j], s[i].c_str());
	      vector<string> w(split(s[i], ' '));
	      date_s = w[0];
      }
    }
  }
  for0(i, 3) if(bi[i] < 0){
    printf("Missing band: %s\n", t[i].c_str());
    err("Missing band");
  }

  dat1 = bread(fn1, nrow, ncol, nband);
  b11 = &dat1[bi[0]];
  b21 = &dat1[bi[1]];
  b31 = &dat1[bi[2]];

  dat2 = bread(fn2, nrow, ncol, nband);
  b12 = &dat2[bi[0]];
  b22 = &dat2[bi[1]];
  b32 = &dat2[bi[2]];

  out = falloc(np * 3);
  for0(i, np){
    out[i]   = (b12[i] - b11[i]) / (b12[i] + b11[i]);
    out[i + np]      = (b22[i] - b21[i]) / (b22[i] + b21[i]);
    out[i + np + np]           = (b32[i] - b31[i]) / (b32[i] + b31[i]);
  }

  vector<str> bn;
  bn.push_back(date_s + str("(b32[i] - b31[i]) / (b32[i] + b31[i]) sentinel2_anomaly.cpp"));
  bn.push_back(date_s + str("(b22[i] - b21[i]) / (b22[i] + b21[i]) sentinel2_anomaly.cpp"));
  bn.push_back(date_s + str("(b12[i] - b11[i]) / (b12[i] + b11[i]) sentinel2_anomaly.cpp"));
  
  hwrite(ohn, nrow, ncol, 3, 4, bn);
  bwrite(out, ofn, nrow, ncol, 3);
  run(str("envi_header_copy_mapinfo.py ") + hfn + str(" ") + ohn);  
  free(dat1);
  free(dat2);
  free(out);
  */
  return 0;
}
