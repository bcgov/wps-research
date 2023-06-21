#include"matrix3.h"
#include"misc.h"

int main(int argc, char ** argv){
  str sep("/");
  vector<str> channels;
  channels.push_back(str("T11.bin")); channels.push_back(str("T12_imag.bin"));
  channels.push_back(str("T12_real.bin")); channels.push_back(str("T13_imag.bin"));
  channels.push_back(str("T13_real.bin")); channels.push_back(str("T22.bin"));
  channels.push_back(str("T23_imag.bin")); channels.push_back(str("T23_real.bin"));
  channels.push_back(str("T33.bin"));

  if(argc < 3) err("t3_change.py [t3 path: pre] [t3 path 2: post]");
  size_t nr, nc, nb, np, i, j;
  vector<float*> a, b;
  float * out;

  for0(i, 9){
    str fn(str(argv[1]) + sep + channels[i]);
    str hfn(hdr_fn(str(argv[1]) + sep + channels[i]));
    hread(hfn, nr, nc, nb);
    a.push_back(bread(fn, nr, nc, nb));

    str fn2(str(argv[2]) + sep + channels[i]);
    str hfn2(hdr_fn(str(argv[2]) + sep + channels[i]));
    hread(hfn2, nr, nc, nb);
    b.push_back(bread(fn2, nr, nc, nb));
  }

  np = nr * nc; // pixels per band
  out = falloc(np * 3); // 3-channel out buffer

  for0(i, np){
    if(i % 1000000 == 0) printf("%f %zu/%zu\n", 100. * (float)(i+1) / (float)np, i+1, np);

    herm3<cf> A;
    herm3<cf> B;
    A.initT3(a[0][i], a[1][i], a[2][i], a[3][i], a[4][i], a[5][i], a[6][i], a[7][i], a[8][i]);
    B.initT3(b[0][i], b[1][i], b[2][i], b[3][i], b[4][i], b[5][i], b[6][i], b[7][i], b[8][i]);
    herm3<cf> C(B - A);

    // matrix3<cf> D(A.inv() *B); // multiplicative version

    vec3<cf> E1, E2, E3, L;
    TYPE R = eig(C, L, E1, E2, E3);  // use D or C here

    out[i] = abs(L.a);
    out[i + np] = abs(L.b);
    out[i + np + np] = abs(L.c);

    if(isnan(abs(L.a)) || isnan(abs(L.b)) || isnan(abs(L.c))) continue;

    if(abs(L.a) != 0. && abs(L.b) != 0. && abs(L.c) != 0.){
      if(!((abs(L.a) > abs(L.b) ) && (abs(L.b) > abs(L.c)))){
        cout << abs(L.a) << endl << abs(L.b) << endl << abs(L.c) << endl;
        cout <<"Error:eigenvectors were not correctly sorted\n";
        exit(1);
      }
    }

  }

  str ofn(str("lambda_") + str(argv[1]) + str("_") + str(argv[2]) + str(".bin"));
  str ohn(str("lambda_") + str(argv[1]) + str("_") + str(argv[2]) + str(".hdr"));
  bwrite(out, ofn, nr, nc, 3);
  hwrite(ohn, nr, nc, 3);

  return 0;
}
