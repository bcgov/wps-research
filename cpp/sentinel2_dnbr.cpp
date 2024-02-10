/* 20230209 sentinel2_dnbr.cpp */
#include"misc.h"

int band_ix(str freq_nm, vector<str> band_names, str file_name){
  /*iterate band names strings to find preferred frequency (nm)*/
  int i;
  for0(i, band_names.size()){
    if(contains(band_names[i], freq_nm)){
      printf("found %s\n", band_names[i].c_str());
      return(i);
    }
  }
  printf("Expected band with frequency: %s in file %s\n", freq_nm.c_str(), file_name.c_str());
  err("exit");
  return -1;
}


int main(int argc, char ** argv){
  if(argc < 3){
    err("sentinel2_dnbr.exe [sentinel2 pre-image] [post-image] (NIR, SWIR req'd)");
  }

  size_t nrow, ncol, nband, nrow2, ncol2, nband2, np, i, j, k, n, ij;
  float * out, * dat1, * dat2, * b1, * b2, * b3;
  long int bi[3];

  str fn_pre(argv[1]); /* binary files */
  str fn_pst(argv[2]); 

  str ofn(fn_pst + "_dnbr.bin");
  str hfn1(hdr_fn(fn_pre)); /* headers */
  str hfn2(hdr_fn(fn_pst));
  str ohn(hdr_fn(ofn, true));

  vector<str> s1;
  vector<str> s2;
  str s_NIR("842nm");
  str s_SWR("2190nm");

  hread(hfn1, nrow, ncol, nband, s1);
  hread(hfn2, nrow2, ncol2, nband2, s2);
  int b08_ix_pre = band_ix(s_NIR, s1, fn_pre);
  int b12_ix_pre = band_ix(s_SWR, s1, fn_pre);
  int b08_ix_pst = band_ix(s_NIR, s2, fn_pst);
  int b12_ix_pst = band_ix(s_SWR, s2, fn_pst);

  if(nrow != nrow2 || ncol != ncol2) err("image pair dimensions must match");
  printf("image dimensions OK\n");

  dat1 = bread(fn_pre, nrow, ncol, nband);
  dat2 = bread(fn_pst, nrow, ncol, nband);
  np = nrow * ncol;
  out = falloc(np);
  
  float nbr_pre, nbr_post;
  float * b08_pre = &dat1[np * b08_ix_pre];
  float * b12_pre = &dat1[np * b12_ix_pre];
  float * b08_pst = &dat2[np * b08_ix_pst];
  float * b12_pst = &dat2[np * b12_ix_pst];

  for0(i, np){
    // NBR = (B08 - B12) / (B08 + B12)
    nbr_pre = (b08_pre[i] - b12_pre[i]) / (b08_pre[i] + b12_pre[i]);
    nbr_pst = (b08_pst[i] - b12_pst[i]) / (b08_pst[i] + b12_pst[i]);
    out[i] = nbr_pre - nbr_pst;
  } 

  hwrite(ohfn, nrow, ncol, 1, 4); // bn
  bwrite(out, ofn, nrow, ncol, 1);

  return 0;
}
