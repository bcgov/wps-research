#include"misc.h"
#include<float.h>

int main(int argc, char ** argv){
  if(argc < 3) err("raster_nearest_centre.cpp [input raster file] [input centres file BIP]");
  str cfn(argv[2]);
  size_t N = 0; // number of centres;


  size_t nrow, ncol, nband, i, j, k, n;

  str bfn(argv[1]); // input "envi type-4" aka IEEE Floating-point 32bit BSQ (band sequential) data stack
  str hfn(hdr_fn(bfn)); // get name of header file
  hread(hfn, nrow, ncol, nband); // get image shape from header
  printf("nrow %d ncol %d nband %d\n", nrow, ncol, nband);
  float * dat = bread(bfn, nrow, ncol, nband); // read image data

  str s;
  vector<str> lines;
  ifstream read(cfn);
  while(std::getline(read, s)){
    cout << s << endl;
    N++;
    lines.push_back(s);
  }
  float * centres = falloc(N);
  for0(i, N){
    centres[i] = atof(lines[i].c_str());
    
  }
  N /= nband;

  
 
  // size_t N = fsize(cfn) / (sizeof(float) * nband);
  printf("find nearest centre of: %zu\n", N);
  size_t np = nrow * ncol;
  float min_i, min_d;

  float * out = falloc(np);
  for0(n, N){
    printf(" %d [", n);
    for0(k, nband){
      printf("%f ",centres[N*n + k]);
    }
    printf("]\n");
  }


  for0(i, nrow){
    size_t ix = i * ncol;
    for0(j, ncol){
        size_t jx = ix + j;
        min_i = NAN;
        min_d = FLT_MAX;
     
        // calculate distance to centre
        for0(n, N){
           float d = 0;
           for0(k, nband){
              float dd = dat[np * k + jx] - centres[N * n + k];
              d += dd * dd;
           }
           d = sqrt(d);
           if(isnan(d) || isinf(d)){
           }
           else{
             if(d < min_d){
                 min_i = n;
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
  free(dat);
  return 0;
}

