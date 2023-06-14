/* 20230613 adapted from htrim2.cpp

"non-proportional" histogram trimming, for hyperspectral imagery with scaling between 0 and 1.. 20210618 bugfix 20210721*/
#include"misc.h"

float N_PERCENT; // histogram percentage to trim

/* new min and max of float array -- n floats */
void p_percent(float * min, float * max, float * dd, size_t n, size_t nb){
  size_t i;
  priority_queue<float> q;

  float mn = FLT_MAX;
  float mx = FLT_MIN;
  
  for0(i, n * nb){
    float d = dd[i];
    if(isinf(d) || isnan(d)){
    }
    else{
      if(d < mn) mn = d;
      if(d > mx) mx = d;
      q.push(d);
    }
  }

  int n_pct = (int)floor(.01 * N_PERCENT * ((float)q.size()));

  for0(i, n_pct) q.pop();
  *max = q.top();
  while(q.size() > n_pct) q.pop();
  *min = q.top();

  printf("mn %e mx %e two_p n=%zu min %e max %e\n",mn, mx, n, *min, *max);
}

int main(int argc, char ** argv){
  if(argc < 2) err("htrim3 [input binary file name] # [optional % trim factor e.g. 1.1]");
  if(argc > 2) N_PERCENT = argc > 2 ? atof(argv[2]): 1.; // default one % trim

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header name
  size_t nrow, ncol, nband, np, i, j, n;
  hread(hfn, nrow, ncol, nband); // read hdr
  np = nrow * ncol; // n input pix

  float * dat = bread(fn, nrow, ncol, nband);
  float * out = falloc(np * nband);
  float mn, mx;

  p_percent(&mn, &mx, dat, np, nband);

  int jx;
  float r, d;
  for0(j, nband){
    jx = np * j;
    r = 1. / (mx - mn);
    for0(i, np){
      d = r * (dat[jx + i] - mn);
      if(d < 0.) d = 0.;
      if(d > 1.) d = 1.;
      out[jx + i] = d;
    }
  }

  str ofn(fn + str("_ht.bin")); // output file
  str ohfn(fn + str("_ht.hdr")); // out hdr
  hwrite(ohfn, nrow, ncol, nband);
  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  n = fwrite(out, sizeof(float), np * nband, f);
  fclose(f);

  free(dat); free(out);
  return 0;
}
