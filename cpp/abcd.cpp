/* 20220517 vim abcd.cpp; clean; python3 compile.py; ./abcd.exe A.bin B.bin C.bin # compare with D.bin
vim abcd.cpp; clean; python3 compile.py; valgrind ./abcd.exe A.bin B.bin C.bin # add g flag for valgrind
Also:
abcd.exe A.bin B.bin A.bin
abcd.exe A.bin A.bin A.bin

NB bin input data using bsq2bip, binary_sort, bip2bsq
2020517 support for C.shape != A.shape */
#include"misc.h"
static size_t nr[3], nc[3], nb[3], skip_f;
static int * bp, * bp2;  // bad px: {A,B}, C
static float * y[3], *x; // {A, B, C}, D
static size_t np, np2;  // npix A, C
size_t m;  // tmp band-ix
float t; // tmp float

void job(size_t i){
  if(bp2[i]) return; // skip bad
  float d;
  size_t j, k;
  size_t mi = 0;
  float md = FLT_MAX;

  for(j = 0; j < np; j += skip_f){ 
    if(bp[j]) continue;
    d = 0;
    for0(k, nb[0])
      d += (y[0][np * k + j] - y[2][np2 * k + i]) * (y[0][np * k + j] - y[2][np2 * k + i]);

    if(d < md){
      md = d;
      mi = j; 
    }
  }

  if(i % 100000 == 0)
    cprint(to_string(100.* ((float)(i+1) / (float)np2)) + str(" % ") + to_string(i) + str(" / ") + to_string(np2));

  for0(k, nb[2])
    x[np2 * k + i] = y[1][np * k + mi];
}

inline int is_bad(float * dat, size_t i, size_t n_b){
  int zero = true;
  for0(m, n_b){  // find bad/empty pix
    t = dat[np * m + i];
    if(isnan(t) || isinf(t)) return true;
    if(t != 0) zero = false;
  }
  return zero;
}

int main(int argc, char** argv){
  if(argc < 5)
  err("view_as [img1 (n bands)] [img2 (m bands)] [img3 (n bands)] [skip]\n");

  skip_f = (size_t) atol(argv[4]);
  printf("skipf %zu\n", skip_f);
  size_t i;

  for0(i, 3)
  hread(hdr_fn(argv[1 + i]), nr[i], nc[i], nb[i]);

  if(nr[0] != nr[1] || nc[0] != nc[1])
    err("dimensions must match");

  if(nb[0] != nb[2])
    err("need same # of bands: images 1, 3");

  x = falloc(nr[2] * nc[2] * nb[2]); // out buf
  for0(i, 3)
    y[i] = bread(str(argv[i + 1]), nr[i], nc[i], nb[i]);

  np = nr[0] * nc[0];
  np2 = nr[2] * nc[2];
  
  bp = ialloc(np);  // bad pix mask
  size_t n_bad = 0;  // flag bad pix
  for0(i, np){
    bp[i] = is_bad(y[0], i, nb[0]) || is_bad(y[1], i, nb[1]);
    if(bp[i]) n_bad ++;
  }
  if(n_bad == np)
    err("no good pix: AxB");

  n_bad = 0;
  for0(i, np2){
    bp2[i] = is_bad(y[3], i, nb[2]);
    if(bp2[i]) n_bad ++;
  }
  if(n_bad == np2)
    err("no good pix: C");

  parfor(0, np2, job);  // for each output result pix (same shape as C)
  str pre(str("view_as_") +
	  str(argv[1]) + str("_") + str(argv[2]) + str("_") +
	  str(argv[3]) + str("_") + str(argv[4]));

  bwrite(x, pre + str(".bin"), nr[2], nc[2], nb[1]);
  hwrite(pre + str(".hdr"), nr[2], nc[2], nb[1]);

  system((str("python3 ~/GitHub/bcws-psu-research/py/raster_plot.py ") +
	 pre + str(".bin 1 2 3 1")).c_str());
  return 0;
}
