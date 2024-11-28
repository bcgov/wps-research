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
  np = nrow * ncol; // now that we are sure the files match, proceed!
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
  return 0;
}
