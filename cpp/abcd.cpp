#include"misc.h" /* 20220517 e.g.: abcd.exe A.bin B.bin C.bin # and compare w D.bin! */
static size_t nr[3], nc[3], nb[3], skip_f, m, np, np2; // shapes
static float * y[3], *x, t, *A, *B, *C;  // data
static int * bp, * bp2;  // bad px: {A,B}, C

void infer_px(size_t i){
  if(bp2[i]) return; // skip bad px in A, B
  float d, e, md = FLT_MAX;
  size_t j, k, mi = 0;
  size_t nb_0 = nb[0];

  for(j = 0; j < np; j += skip_f){  // uniform sample in space
    if(bp[j]) continue;  // skip bad px in C
    d = 0;
    for0(k, nb_0){
      e = A[np * k + j] - C[np2 * k + i];
      d += e * e;
    }
    if(d < md)
      (md = d, mi = j);  // nearer
  }
  for0(k, nb[2])
    x[np2 * k + i] = B[np * k + mi];  // assign nearest
  if(i % 100000 == 0) status(i, np2);
}

inline int is_bad(float * dat, size_t i, size_t n_b){
  int zero = true;
  for0(m, n_b){  // find bad/empty pix
    t = dat[np * m + i];
    if(isnan(t) || isinf(t)) return true;
    if(t != 0) zero = false;
  }
  return (n_b > 1 && zero); // 0 in 1-band product isn't bad
}

int main(int argc, char** argv){
  size_t i, n_bad;
  if(argc < 4)
    err("abcd [img1 (n bands)] [img2 (m bands)] [img3 (n bands)] # [skip]\n");
  skip_f = (argc > 4) ? (size_t) atol(argv[4]): 1; // bsq2bip -> binary_sort -> bip2bsq

  for0(i, 3)
    hread(hdr_fn(argv[1 + i]), nr[i], nc[i], nb[i]);
  if(nr[0] != nr[1] || nc[0] != nc[1])
    err("A.shape != B.shape");
  if(nb[0] != nb[2])
    err("A.n_bands != C.n_bands");
  (np = nr[0] * nc[0], np2 = nr[2] * nc[2]);
  if(skip_f >= np)
    err("illegal skip_f");

  x = falloc(nr[2] * nc[2] * nb[2]); // out buf
  for0(i, 3)
    y[i] = bread(str(argv[i + 1]), nr[i], nc[i], nb[i]);  // read input

  (n_bad = 0, bp = ialloc(np));  // bad pixels in A, B?
  for0(i, np){
    bp[i] = is_bad(y[0], i, nb[0]) || is_bad(y[1], i, nb[1]);
    if(bp[i]) n_bad ++;
  }
  if(n_bad == np)
    err("no good pix: AxB");

  (n_bad = 0, bp2 = ialloc(np2));  // bad pxls in C?
  for0(i, np2){
    bp2[i] = is_bad(y[2], i, nb[2]);
    if(bp2[i]) n_bad ++;
    x[i] = NAN;
  }
  if(n_bad == np2)
    err("no good pix: C");
  (A = y[0], B = y[1], C = y[2]);

  str u("_");
  parfor(0, np2, infer_px);  // inference by output pixel

  str pre(str("abcd_") + str(argv[1]) + u + str(argv[2]) + u +
		         str(argv[3]) + u + str(argv[4]));

  bwrite(x, pre + str(".bin"), nr[2], nc[2], nb[1]);  // write out
  hwrite(pre + str(".hdr"), nr[2], nc[2], nb[1]);

  system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre +
	  str(".bin 1 2 3 1")).c_str());
  return 0;
}
