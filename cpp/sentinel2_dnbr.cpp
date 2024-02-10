/* 20230209 sentinel2_dnbr.cpp */
#include"misc.h"

int band_ix(str freq_nm, vector<str> band_names, str file_name){
  int i; /* find index of the band with wavelength freq_nm in file file_name*/
  for0(i, band_names.size()) if(contains(band_names[i], freq_nm)) return(i);
  printf("Expected band at freq.: %s in %s\n", freq_nm.c_str(), file_name.c_str());
  err("exit");
  return -1;
}

int main(int argc, char ** argv){
  if(argc < 3) err("sentinel2_dnbr [s2 preimg] [s2 post] [opt'l flag: writeNBR] # NIR, SWR rq'd");
  float nbr_pre, nbr_pst, *out, *dat1, *dat2, *b1, *b2, *b3, *out2, *out3;
  size_t nrow, ncol, nband, nrow2, ncol2, nband2, np, i, j, k, n, ij;
  long int bi[3];

  str fn_pre(argv[1]), fn_pst(argv[2]); /* ENVI format data to read */
  str hfn1(hdr_fn(fn_pre)), hfn2(hdr_fn(fn_pst)); /* headers */
  str s_NIR("842nm"), s_SWR("2190nm"); /* rq'd wavelengths */
  vector<str> s1, s2; /* two lists of band names */
  
  str ofn(fn_pst + "_dnbr.bin"), ofn2(fn_pre + "_nbr.bin"), ofn3(fn_pst + "_nbr.bin");
  str ohn(hdr_fn(ofn, true)), ohn2, ohn3; /* output files + headers */
  
  (hread(hfn1, nrow, ncol, nband, s1), hread(hfn2, nrow2, ncol2, nband2, s2));
  int b08_ix_pre = band_ix(s_NIR, s1, fn_pre), b12_ix_pre = band_ix(s_SWR, s1, fn_pre);
  int b08_ix_pst = band_ix(s_NIR, s2, fn_pst), b12_ix_pst = band_ix(s_SWR, s2, fn_pst);

  if(nrow != nrow2 || ncol != ncol2) err("image pair dimensions must match");
  (dat1 = bread(fn_pre, nrow, ncol, nband), dat2 = bread(fn_pst, nrow, ncol, nband));
  np = nrow * ncol;
  out = falloc(np);

  if(argc > 3){
    (ohn2 = hdr_fn(fn_pre + "_nbr.hdr", true), ohn3 = hdr_fn(fn_pst + "_nbr.hdr", true));
    (out2 = falloc(np), out3 = falloc(np));
  }
  
  float * b08_pre = &dat1[np * b08_ix_pre], * b12_pre = &dat1[np * b12_ix_pre];
  float * b08_pst = &dat2[np * b08_ix_pst], * b12_pst = &dat2[np * b12_ix_pst];
  
  for0(i, np){
    // NBR = (B08 - B12) / (B08 + B12)
    nbr_pre = (b08_pre[i] - b12_pre[i]) / (b08_pre[i] + b12_pre[i]);
    nbr_pst = (b08_pst[i] - b12_pst[i]) / (b08_pst[i] + b12_pst[i]);
    if(argc > 3) (out2[i] = nbr_pre, out3[i] = nbr_pst);
    out[i] = nbr_pre - nbr_pst;
  } 

  (hwrite(ohn, nrow, ncol, 1, 4), bwrite(out, ofn, nrow, ncol, 1));

  if(argc > 3){
    (hwrite(ohn2, nrow, ncol, 1, 4), bwrite(out2, ofn2, nrow, ncol, 1));
    (hwrite(ohn3, nrow, ncol, 1, 4), bwrite(out3, ofn3, nrow, ncol, 1)); 
  }
  (free(out), free(out2), free(out3), free(dat1), free(dat2));
  return 0;
}
