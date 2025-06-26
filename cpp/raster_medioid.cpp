/* 20250626 compute medioid for a list of rasters. Use random file access, to avoid loading the whole files in memory */

#include"misc.h"

size_t nrow, ncol, nband, np, T;
FILE ** infiles; // input file pointers for random access
float * out; // output buffer
pthread_mutex_t print_mutex; // mutex for printing

// calculate medioid for pixel j
void medioid(size_t j){
  if(j % 1000 == 0){
    pthread_mutex_lock(&print_mutex);
    printf("processing pixel %zu of %zu\n", j + 1, np);
    pthread_mutex_unlock(&print_mutex);
  }

  vector<vector<float>> data(T);
  FILE * f;
  int i, k;
  for0(i, T){
    f = infiles[i];
    data[i] = vector<float>(nband);
    for0(k, nband){
      fseek(f, np * k + j, SEEK_SET);
      fread(&data[i][k], sizeof(float), 1, f);
    }
    //cout << data[i] << endl;
  }
}

int main(int argc, char ** argv){
  size_t i, j, np, k, nrow2, ncol2, nband2;

  if(argc < 4){
    err("raster_medioid [raster file 1] .. [raster file N] [output file]");
  }

  T = argc - 2;
  FILE * outfile = wopen(argv[argc-1]);
  
  infiles = (FILE **)(void *)malloc(sizeof(FILE *) * T);
  if(!infiles) err("malloc failed");
  memset(infiles, 0, sizeof(FILE *) * T);

  for(i = 0; i < T; i++){
    infiles[i] = ropen(argv[i + 1]);
    printf("+r %s\n", argv[i + 1]);
    if(infiles[i]==NULL) err("failed to open input file");

    str hfn(hdr_fn(str(argv[i + 1])));  // input header file name
    if(i==0){
      hread(hfn, nrow, ncol, nband);
      np = nrow * ncol; // number of pixels
    }
    else{
      hread(hfn, nrow2, ncol2, nband2);
      if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
        err(str("file: ") + str(argv[i + 1]) + str(" has different dimenions than ") + str(argv[1]));
      }
    }
  }

  out = falloc(np);
  for0(j, np){
    cout << j + 1 << " " << np << endl;
    medioid(j);
  }
  str ofn(argv[argc-1]);
  str ohfn(hdr_fn(ofn, true));
  bwrite(out, ofn, nrow, ncol, nband);
  run((str("cp -v ") + hdr_fn(str(argv[1])) + str(" ") + ohfn).c_str());

  fclose(outfile);
  for(i = 0; i < T; i++) {
    if(infiles[i]) fclose(infiles[i]);
  }
  free(infiles);
  return 0;
}


