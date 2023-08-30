/* 20230527 calculate stokes parameters from C2 matrix (compact pol)
We output C2 matrix from SNAP using snap2psp.py after speckle filtering

** N.B. assumed RHC transmit ** Should have a "named variables" class for writing files automatically from named variables


[1] IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, VOL. 9, NO. 1, JANUARY 2012
"Compact Decomposition Theory" , S. R. Cloude, Fellow, IEEE, D. G. Goodenough, Fellow, IEEE, and H. Chen

[2]  Hao Chen, Joanne C. White & André Beaudoin (2023) Derivation and
assessment of forest-relevant polarimetric indices using RCM compact-pol data, International
Journal of Remote Sensing, 44:1, 381-406, DOI: 10.1080/01431161.2022.2164528
*/

#include"misc.h"

void copy_header(str in_dir, str * fn, str var_name){
    str cmd(str("cp -v ") + hdr_fn(in_dir + str("/") + fn[0]) + str(" ") + in_dir + str("/") + var_name + str(".hdr"));
    printf("%s\n", cmd.c_str());
    system(cmd.c_str());
}

int main(int argc, char ** argv){
  if(argc < 2){
    err("c2_stokes [C2 input directory: polsarpro format .bin with .hdr");
  }
  str in_dir(argv[1]);
  size_t nrow, ncol, nband, np, i;

  str fn[4] = {str("C11.bin"), str("C12_real.bin"), str("C12_imag.bin"), str("C22.bin")};
  str hfn(hdr_fn(in_dir + str("/") + fn[0]));
  hread(hfn, nrow, ncol, nband); // read header file for ENVI type 6 input file
  np = nrow * ncol; // number of pixels
  if(nband != 1) err("expected: 1-band file");

  float ** C2 = (float**)alloc(sizeof(float*) * 4); // load input
  for0(i, 4) C2[i] = bread(in_dir + str("/") + fn[i], nrow, ncol, nband);
  
  float ** g = (float**)alloc(sizeof(float*) * 4); // output buffer
  for0(i, 4) g[i] = falloc(np);

  float * m = falloc(np); // degree of polarisation
  float * delta = falloc(np);  // phase between linear components of wave
  float * alpha_s = falloc(np);
  float * phi = falloc(np);
  float * rvog_m_v = falloc(np);
  float * rvog_m_s = falloc(np);
  float * rvog_alpha_s = falloc(np);
  float * p_d = falloc(np);
  float * p_v = falloc(np);
  float * p_s = falloc(np);
  float * rvi = falloc(np);
  float * csi = falloc(np);
  float * rfdi = falloc(np);

  #define C11 0
  #define C12_re 1
  #define C12_im 2
  #define C22 3
  for0(i, np){
    g[0][i] = C2[C11][i] + C2[C22][i]; // eqn (2) from [1]
    g[1][i] = 2* C2[C11][i] - g[0][i]; // eqn (2) from [1]
    g[2][i] = C2[C12_re][i]; // eqn (2) from [1]
    g[3][i] = C2[C12_im][i]; // eqn (2) from [1]

    double g12g22 = (double)g[1][i] * (double)g[1][i] + (double)g[2][i] * (double)g[2][i];
    m[i] = (float)(pow(g12g22 + (double)g[3][i] * (double)g[3][i], .5) / (double)g[0][i]); // eqn (4) from [1]
    delta[i] = (float)atan((double)g[3][i] / (double)g[2][i]);  // eqn. (4) from [1]

    alpha_s[i] = 0.5 * atan(pow(g12g22, .5) / (- g[3][i]));  // - sign for RHC transmit?
    phi[i] = atan2(-g[2][i], g[1][i]);  // - sign for RHC xmit?

    // H_w should go in here from [1]

    // rvog model [1]
    rvog_m_v[i] = .5 * g[0][i] * ( 1. - m[i]);
    rvog_m_s[i] = 2. * g[0][i] * m[i];
    rvog_alpha_s[i] = 0.5 * (float)atan(pow(g12g22, .5) / (double)g[3][i]);

    // pseudo 3-component decomp [1]
    p_d[i] = .5 * g[0][i] * m[i] * (1. - (float)cos(2. * (double)rvog_alpha_s[i]));
    p_v[i] = g[0][i] * (1. - m[i]);
    p_s[i] = .5 * g[0][i] * m[i] * (1 + (float)cos(2. * (double)rvog_alpha_s[i]));

    // compact-pol parameters: eq'n 18 from [2] 
    rvi[i] = 1 - m[i];
    float r = g[2][i] / g[1][i];
    csi[i] = (4. * r) / (3. + m[i]);
    rfdi[i] = 2. * (r + m[i]) / (3. + 2. * r - m[i]);

  }   

  for0(i, 4)
    bwrite(g[i], in_dir + str("/g") + to_string(i + 1) + str(".bin"), nrow, ncol, nband);
  
  bwrite(m, in_dir + str("/m.bin"), nrow, ncol, nband);
  bwrite(delta, in_dir + str("/delta.bin"), nrow, ncol, nband);
  bwrite(alpha_s, in_dir + str("/alpha_s.bin"), nrow, ncol, nband);
  bwrite(phi, in_dir + str("/phi.bin"), nrow, ncol, nband);
  bwrite(rvog_m_v, in_dir + str("/rvog_m_v.bin"), nrow, ncol, nband);
  bwrite(rvog_m_s, in_dir + str("/rvog_m_s.bin"), nrow, ncol, nband);
  bwrite(rvog_alpha_s, in_dir + str("/rvog_alpha_s.bin"), nrow, ncol, nband);
  bwrite(p_d, in_dir + str("/p_d.bin"), nrow, ncol, nband);
  bwrite(p_v, in_dir + str("/p_v.bin"), nrow, ncol, nband);
  bwrite(p_s, in_dir + str("/p_s.bin"), nrow, ncol, nband);
  bwrite(rvi, in_dir + str("/rvi.bin"), nrow, ncol, nband);
  bwrite(csi, in_dir + str("/csi.bin"), nrow, ncol, nband);
  bwrite(rfdi, in_dir + str("/rfdi.bin"), nrow, ncol, nband);

   for0(i, 4){
    copy_header(in_dir, fn, str("g") + to_string(i + 1));
  }

  copy_header(in_dir, fn, str("m"));
  copy_header(in_dir, fn, str("delta"));
  copy_header(in_dir, fn, str("alpha_s"));
  copy_header(in_dir, fn, str("phi"));
  copy_header(in_dir, fn, str("rvog_m_v"));
  copy_header(in_dir, fn, str("rvog_m_s"));
  copy_header(in_dir, fn, str("rvog_alpha_s"));
  copy_header(in_dir, fn, str("p_d"));
  copy_header(in_dir, fn, str("p_v"));
  copy_header(in_dir, fn, str("p_s")); 
  copy_header(in_dir, fn, str("rvi"));
  copy_header(in_dir, fn, str("csi"));
  copy_header(in_dir, fn, str("rfdi"));


  for0(i, 4) (free(C2[i]), free(g[i]));
  free(C2);
  free(g);
  free(m);
  free(delta);
  free(alpha_s);
  free(phi);
  free(rvog_m_v);
  free(rvog_m_s);
  free(rvog_alpha_s);
  free(p_d);
  free(p_v);
  free(p_s);
  return 0;
}
