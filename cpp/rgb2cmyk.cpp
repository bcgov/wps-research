/* 20230617 rgb2cmyk: convert RGB to CMYK color model */
int main(int argc, char ** argv){
  if(argc < 2) err("rgb2cmyk [input ENVI-format raster file name, 3 band]");

  str fn(argv[1]); // input image file name
  if(!exists(fn)) err(str("failed to open input file: ") + fn);
  str hfn(hdr_fn(fn)); // input header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  if(nband != 3) err("3 band input supported"); // need rgb

  size_t i, j, k, m;
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(nrow * ncol * (nband + 1)); // four channels for output

  float r, g, b, C, M, Y, K;
  for0(i, nrow){
    for0(j, ncol){
      k = i * ncol + j;
      r = dat[k];
      g = dat[k + np];
      b = dat[k + np + np];

      rgb2cmyk(r, g, b, *C, *M, *Y, *K);

      m = k;
      out[m] = C; m += np;
      out[m] = M; m += np;
      out[m] = Y; m += np;
      out[m] = K;      
    }
  }


  str ofn(fn + str("_cmyk.bin"));
  str ohn(fn + str("_cmyk.hdr"));

  str sep(" ");
  bwrite(out, ofn, nrow, ncol, nband);
  str cmd(str("cp -v ") + hfn + sep + ohn);
  cout << cmd << endl;
  int a = run(cmd);
  // envi_header_modify.py [.hdr file to modify] [nrow] [ncol] [nband] [band 1 name]... [band n name]

  cmd = str("envi_header_modify.py ") + ohn + sep + to_string(nrow) + sep + to_string(ncol) + sep + to_string((int)4) + str(" C M Y K");
  a = run(cmd);

  return 0;
}



