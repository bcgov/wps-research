#include"misc.h"
#include<float.h>

int main(int argc, char ** argv){
  float * centres, *out, *dat, min_i, min_d, d, dd;
  size_t N = 0, nrow, ncol, nband, i, j, k, n, ix, jx, np;
  if(argc < 3) err("raster_nearest_centre.cpp [input raster file] [input centres file newline-separated-value]");

  str cfn(argv[2]);
  str bfn(argv[1]); // input "envi type-4" bsq
  str hfn(hdr_fn(bfn)); // get name of header file
  hread(hfn, nrow, ncol, nband); // get shape from header
  printf("nrow %d ncol %d nband %d\n", nrow, ncol, nband);
  dat = bread(bfn, nrow, ncol, nband); // read image data

  str s;
  vector<str> lines;
  ifstream read(cfn);
  while(std::getline(read, s)){
    trim(s);
    if(s != str(""))
    lines.push_back(s);
  }
  N = lines.size();
  centres = falloc(N);
  printf("N %zu\n", N);
  for0(i, N) centres[i] = atof(lines[i].c_str());

  N /= nband;
  printf("N %zu\n", N);
  printf("find nearest centre of: %zu\n", N);

  np = nrow * ncol;
  out = falloc(np);
  for0(n, N){
    printf(" %d [", n);
    for0(k, nband) printf("%f ",centres[n * nband + k]);
    printf("]\n");
  }

  for0(i, nrow){
    ix = i * ncol;
    for0(j, ncol){

        jx = ix + j;
        min_i = NAN;
        min_d = FLT_MAX;

        for0(n, N){
           d = 0.; // dist to centre
           for0(k, nband){
              dd = dat[np * k + jx] - centres[n * nband + k];
              d += dd * dd;
           }
           d = sqrt(d);
           if(isnan(d) || isinf(d)){
           }
           else{
             if(d < min_d){
                 min_i = (float)n;
                 min_d = d;
               }
           }
        }
        out[jx] = min_i;
    }
  }
  
  bwrite(out, bfn + str("_nearest_centre.bin"), nrow, ncol, 1);
  hwrite(     bfn + str("_nearest_centre.hdr"), nrow, ncol, 1);
  free(centres);
  free(out);
  free(dat);
  return 0;
}
