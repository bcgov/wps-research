/* 20210220 from a raster, select the most common value in a window */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("mode_filter [input binary file] [window size]");

  size_t ws = atoi(argv[2]);
  if(ws % 2 != 1) err("window size must be odd number");
  int dw = (ws - 1) / 2;

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, k, n;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  if(nband != 1) err("defined for 1-band images");
  unordered_map<float, size_t> values;
  unordered_map<float, size_t>::iterator it;

  float d, * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(nrow * ncol * sizeof(float));
  long int ix, iy, di, dj, dx, dy, i, j;
 
  for0(i, nrow){
    for0(j, ncol){
      ix = i * ncol + j;
      out[ix] = dat[ix];

      values.clear();

      for(di = -dw; di <= dw; di++){
        dx = i + di;
        if(dx > nrow || dx < 0) continue;
        ix = dx * ncol;

        for(dj = -dw; dj <= dw; dj++){
          dy = j + dj;
          if(dy > ncol || dy < 0) continue;

          d = dat[ix + dy];
          if(!(isinf(d) || isnan(d))){
            if(values.count(d) < 1) values[d] = 0;
            values[d] += 1;
          }
        }
      }

      if(values.size() > 0){
        d = values.begin()->first;
        k = values.begin()->second;
        for(it = values.begin(); it != values.end(); it++){
          if(it->second > k){
            d = it->first;
            k = it->second;
          }
        }
        out[i * ncol + j] = d;
      }
    }
  }

  str ofn(fn + str("_mode_filter.bin")); // write output file
  str ohfn(fn + str("_mode_filter.hdr"));
  hwrite(ohfn, nrow, ncol, nband); // write output header
  cout << "+w " << ofn << endl;

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(out, sizeof(float) * nrow * ncol, 1, f); // write data
  fclose(f);

  free(dat);
  free(out);
  return 0;
}
