/* "non-proportional" histogram trimming, for hyperspectral imagery 20210618 */
#include"misc.h"

float N_PERCENT;

/* min and max of float array -- n floats */
void p_percent(float * min, float * max, float * dd, int n){
  size_t i;
  priority_queue<float> q;

  for0(i, n){
    float d = dd[i];
    if(isinf(d) || isnan(d)) continue;
    else q.push(d);
  }

  int n_pct = (int)floor(.01 * N_PERCENT * ((float)q.size()));

  for0(i, n_pct)
    q.pop();
  *max = q.top();
  
  while(q.size() > n_pct)
    q.pop();
  *min = q.top();
 
 printf("two_p n=%zu min %f max %f\n", n, min, max);
}

int main(int argc, char ** argv){
  if(argc < 2) err("htrim2 [input binary file name] # [optional percentage trim factor e.g. 1.1]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  if(argc > 2) N_PERCENT = argc > 2 ? atof(argv[2]): 1.; // default one % trim both sides..

  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(np * nband);

  float * mn = falloc(nband);
  float * mx = falloc(nband);

  for0(i, nband)
    p_percent(&mn[i], &mx[i], &dat[i *np], np);


  int jx;
  float jmn, jmx, r;
  for0(j, nband){
    jx = np * j;
    jmn = mn[j]; jmx = mx[j];
    r = 1. / (jmx - jmn);
    for0(i, np) out[jx + i] = (dat[jx + i] - jmn) * r;
  }

  // write output file
  str ofn(fn + str("_ht.bin"));
  str ohfn(fn + str("_ht.hdr"));
  hwrite(ohfn, nrow, ncol, nband); // write output header
  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  for0(i, np * nband)
    fwrite(out, sizeof(float), np * nband, f); // write data
  fclose(f);
  free(dat);
  free(out);
  free(mn);
  free(mx);
  return 0;
}




  

