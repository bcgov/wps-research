/* 20250626 compute medioid for a list of rasters. Use random file access, to avoid loading the whole files in memory */
#include"misc.h"
#include <cmath>
#include <algorithm>
#include <limits>

size_t nrow, ncol, nband, np, T;
FILE ** infiles; // input file pointers for random access
float ** dat; // input buffer
float * out; // output buffer
pthread_mutex_t print_mutex; // mutex for printing

// calculate medioid for pixel j
void medoid(size_t j){
  if(j % 100000 == 0){
    pthread_mutex_lock(&print_mutex);
    printf("processing pixel %zu of %zu %f/100.\n", j + 1, np, 100. * (float)(j+1) / ((float)np) );
    pthread_mutex_unlock(&print_mutex);
  }

  int i, k, t;
  // Compute medoid index and vector with NaN tolerance (median inlined)
  //int compute_medoid_with_nan(const std::vector<std::vector<float>>& data, std::vector<float>& medoid_out) {
  std::vector<float> median(nband);
  float d;
  // Inline median computation per band
  std::vector<float> valid_values;
  for0(k, nband){
    valid_values.clear();
    for0(t, T){
      d = dat[t][np * k + j];
      if(!std::isnan(d)){ 
        //data[t][b])){
        valid_values.push_back(d); //data[t][b]);
      }
    }

    if(valid_values.empty()){
      median[k] = NAN;
    }
    else{
      std::sort(valid_values.begin(), valid_values.end());
      int n = valid_values.size();
      median[k] = (n % 2 == 0)
        ? (valid_values[n / 2 - 1] + valid_values[n / 2]) / 2.0f
        : valid_values[n / 2];
    }
  }

  bool median_only = true;

  if(median_only){
    // cout << "median " << median << endl;
    for0(k, nband){
      // float mk = median[k];
      out[k * np + j] =  median[k];
    }
    return;
  }
  else{


  // Find medoid: vector closest to the median (NaN-tolerant distance)
  int medoid_index = -1;
  float min_dist = std::numeric_limits<float>::infinity();

  for(int t = 0; t < T; ++t){
    //const std::vector<float>& vec = data[t];
    float sum = 0.0f;
    int valid_count = 0;

    for0(k, nband){
      d = dat[t][np * k + j];
      if(!isnan(d) && !isnan(median[k])) {
        float dd = d - median[k];
        sum += dd * dd;
        ++valid_count;
      }
    }

    float dist = (valid_count > 0) ? sum / valid_count : std::numeric_limits<float>::infinity();
    if(dist < min_dist){
      min_dist = dist;
      medoid_index = t;
    }
  }

  // Return the medoid vector
  if (medoid_index >= 0) {
    // cout << "medoid " << j << " " << data[medoid_index] << endl;
    for0(k, nband) out[k * np + j] = dat[medoid_index][k];
  }
  else{
    for0(k, nband) out[k * np + j] = NAN;
  }

  }
}

int main(int argc, char ** argv){
  size_t i, j, nrow2, ncol2, nband2;
  pthread_mutex_init(&print_mutex, NULL);

  if(argc < 4){
    err("raster_medoid [raster file 1] .. [raster file N] [output file]");
  }

  T = argc - 2;
  FILE * outfile = wopen(argv[argc-1]);
  
  infiles = (FILE **)(void *)malloc(sizeof(FILE *) * T);
  if(!infiles) err("malloc failed");
  memset(infiles, 0, sizeof(FILE *) * T);
  
  dat = (float **)(void *)malloc(sizeof(float *) * T);
  memset(dat, 0, sizeof(float *) * T);

  for(i = 0; i < T; i++){
    // infiles[i] = ropen(argv[i + 1]);
    //printf("+r %s\n", argv[i + 1]);
    //if(infiles[i]==NULL) err("failed to open input file");

    str hfn(hdr_fn(str(argv[i + 1])));  // input header file name
    if(i==0){
      hread(hfn, nrow, ncol, nband);
      np = nrow * ncol; // number of pixels
    }
    else{
      hread(hfn, nrow2, ncol2, nband2);
      if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
        err(str("file: ") + str(argv[i + 1]) + str(" has different shape than ") + str(argv[1]));
      }
    }
    np = nrow * ncol;
  }

  for0(i, T){
    printf("+r %s\n", argv[i + 1]);
    dat[i] = bread(argv[i + 1], nrow, ncol, nband);
  }

  
  // allocate output product area 
  out = falloc(np * nband);
  if(false){
    for0(j, np) medoid(j);
  }
  else{
    parfor(0, np-1, medoid);
  }

  str ofn(argv[argc-1]);
  str ohfn(hdr_fn(ofn, true));
  bwrite(out, ofn, nrow, ncol, nband);
  run((str("cp -v ") + hdr_fn(str(argv[1])) + str(" ") + ohfn).c_str());

  fclose(outfile);
  for(i = 0; i < T; i++) {
    // if(infiles[i]) fclose(infiles[i]);
    free(dat[i]);
  }
  free(infiles);
  free(dat);
  return 0;
}


