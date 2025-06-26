/* 20250626 compute medioid for a list of rasters. Use random file access, to avoid loading the whole files in memory */

#include"misc.h"

size_t nrow, ncol, nband, np, n_files;
FILE ** infiles; // input file pointers for random access
float * out; // output buffer

// calculate medioid for pixel j
void medioid(size_t j){
  vector<vector<float>> data(n_files);
  FILE * f;
  int i, k;
  for0(i, n_files){
    f = infiles[i];
    data[i] = vector<float>(n_bands);
    for0(k, n_bands){
      fseek(f, np * k + j, SEEK_SET);
      fread(&d[k], sizeof(float), 1, f);
    }
    cout << data[i] << endl;
  }
}

int main(int argc, char ** argv){
  size_t i, j, np, k, nrow2, ncol2, nband2;

  if(argc < 4){
    err("raster_medioid [raster file 1] .. [raster file N] [output file]");
  }

  n_files = argc - 2;
  FILE * outfile = wopen(argv[argc-1]);
  
  infiles = (FILE **)(void *)malloc(sizeof(FILE *) * n_files);
  if(!infiles) err("malloc failed");
  memset(infiles, 0, sizeof(FILE *) * n_files);

  for(i = 0; i < n_files; i++){
    infiles[i] = ropen(argv[i + 1]);
    printf("+r %s\n", argv[i + 1]);
    if(infiles[i]==NULL) err("failed to open input file");

    str hfn(hdr_fn(str(argv[i + 1]));  // input header file name
    if(i==0){
      hread(hfn, nrow, ncol, nband);
      np = nrow * ncol; // number of pixels
    }
    else{
      hread(hfn, nrow2, ncol2, nband2);
      if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
        err(str("file: ") + fn + str(" has different dimenions than ") + lines[0]);
      }
    }
  }







  fclose(outfile);
  for(i = 0; i < n_files; i++) {
    if(infiles[i]) fclose(infiles[i]);
  }
  free(infiles);
  return 0;
}


