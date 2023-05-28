/* 20230527 calculate stokes parameters from C2 matrix (compact pol)
We output C2 matrix from SNAP using snap2psp.py after speckle filtering

** N.B. assumed RHC transmit ** 

Should have a "named variables" class for writing files automatically from named variables */
#include"misc.h"

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

    alpha_s[i] = 0.5 * atan(pow(g12g22, .5) / (- g[3][i]));  // - sign for RHC transmit
    phi[i] = atan2(-g[2][i], g[1][i]);  // - sign for RHC xmit

  }   

  for0(i, 4)
    bwrite(g[i], in_dir + str("/g") + to_string(i + 1) + str(".bin"), nrow, ncol, nband);
  
  bwrite(m, in_dir + str("/m.bin"), nrow, ncol, nband);
  bwrite(delta, in_dir + str("/delta.bin"), nrow, ncol, nband);

   for0(i, 4){
    str cmd(str("cp -v ") + hdr_fn(in_dir + str("/") + fn[0]) + str(" ") + in_dir + str("g") + to_string(i + 1) + str(".hdr"));
    printf("%s\n", cmd.c_str());
    system(cmd.c_str());
  }
  {
    str cmd(str("cp -v ") + hdr_fn(in_dir + str("/") + fn[0]) + str(" ") + in_dir + str("/m.hdr"));
    printf("%s\n", cmd.c_str());
    system(cmd.c_str());
  }
  {
    str cmd(str("cp -v ") + hdr_fn(in_dir + str("/") + fn[0]) + str(" ") + in_dir + str("/delta.hdr"));
    printf("%s\n", cmd.c_str());
    system(cmd.c_str());
  }

  for0(i, 4) (free(C2[i]), free(g[i]));
  free(C2);
  free(g);
  free(m);
  free(delta);
  return 0;
}
