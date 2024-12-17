/* 20241128 sentinel2_redmax.cpp find the "reddest" pixels in a sentinel-2 sequence
Caveat:  Assume that the longest wavelength ( B12 ) is first! */
#include"misc.h"
int main(int argc, char ** argv){
  int a = system("rm sentinel2_redmax.bin tmp*bin");

  vector<str> lines( split(exec("ls -1 *.bin"), '\n'));

  size_t i, j, nrow, ncol, nband, np, k, nrow2, ncol2, nband2;

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
      for0(j, np){
        float red1 = 0.;
        float red2 = 0.;
        bool new_good = true;
        bool old_good = true;
        for0(k, nband){
          if(isnan(dat[np*k + j])){
              new_good = false;
          }
          if(isnan(out[np*k + j])){
            old_good = false;
          }
          red1 += out[np * k + j]; // add up the bands, this pixel ( candidate result so far ) 
          red2 += dat[np * k + j]; // current image being processed, this pixel 
        }

        if(!new_good) continue;  // use the new data

        red1 = out[j] / red1;  // red as fraction of sum of bands
        red2 = dat[j] / red2;  // red as fraction of sum of bands
        if(red2 > 1.5 * red1 || (!old_good)){
          for0(k, nband) out[np * k + j]= dat[np * k + j];
        }
      }
    }
    // update the index of the file considered ( i only used to check if on the first pixel ) 
    i++;
    free(dat);   
  }

  bwrite(out, str("sentinel2_redmax.bin"), nrow, ncol, nband);
  run((str("cp -v ") + hdr_fn(lines[0]) + str(" sentinel2_redmax.hdr")).c_str());

  free(out);
  return 0;
}
