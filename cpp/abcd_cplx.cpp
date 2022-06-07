#include"misc.h" /* 20220606 this adapts abcd.cpp for multi-band complex (ENVI type 6) "iq" format data
   20220517 e.g.: abcd.exe A.bin B.bin C.bin # and compare w D.bin!
NB: the output result is complex, as well.
Need:  convert_float_to_cplx.exe, convert_cplx_to_float.exe to prepare class maps,
and extract class maps (afterwards) respectively. */
static size_t nr[3], nc[3], nb[3], skip_f, m, np, np2; // shapes
static float * y[3], *x, t, u, *A, *B, *C;  // data
static int * bp, * bp2;  // bad px: {A,B}, C

void infer_px(size_t i){
  if(bp2[i]) return; // skip bad px in A, B
  float d, e, f, md = FLT_MAX;
  size_t ix1, ix2, j, k, mi = 0;
  size_t nb_0 = nb[0];

  for(j = 0; j < np; j += skip_f){  // uniform sample in space
    if(bp[j]) continue;  // skip bad px in C
    d = 0;
    for0(k, nb_0){
      (ix1 = 2 * (np * k + j)), (ix2 = 2 * (np2 * k + i)); // e = A[np * k + j] - C[np2 * k + i];
      (e = A[ix1] - C[ix2]), (f = A[ix1 + 1] - C[ix2 + 1]);
      d += (float) sqrt((double)(e * e + f * f));
    }
    if(d < md)
      (md = d, mi = j);  // nearer
  }
  for0(k, nb[2])
    x[np2 * k + i] = B[np * k + mi];  // assign nearest
  if(i % 100000 == 0) status(i, np2);
}

inline int is_bad(float * dat, size_t i, size_t n_b){
  size_t ix;
  int zero = true;
  for0(m, n_b){  // find bad/empty pix
    ix = 2 * (np * m + i);
    (t = dat[ix]), (u = dat[ix + 1]);
    if(isnan(t) || isinf(t) || isnan(u) || isinf(u)) return true;
    if(t != 0. || u != 0.) zero = false;
  }
  return (n_b > 1 && zero); // 0 in 1-band product isn't bad
}

int main(int argc, char** argv){
  size_t i, n_bad, n_rd[3];
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

  x = falloc(nr[2] * nc[2] * nb[2] * 2); // out buf
  for0(i, 3){
    y[i] = float_read(str(argv[i + 1]), n_rd[i]); // y[i] = bread(str(argv[i + 1]), nr[i], nc[i], nb[i]);  // read input
    if(n_rd[i] != 2 * nr[i] * nc[i]){
      err("unexpected float count for type-6 (iq 32bit each)");
    }
  }
  (n_bad = 0, bp = ialloc(np));  // bad pixels in A, B?
  for0(i, np){
    bp[i] = is_bad(y[0], i, nb[0]) || is_bad(y[1], i, nb[1]);
    if(bp[i]) n_bad ++; // GOT HERE !!!!!!!!!!!
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

  str us("_");
  parfor(0, np2, infer_px);  // inference by output pixel

  str pre(str("abcd_cplx_") + str(argv[1]) + us + str(argv[2]) + us +
		         str(argv[3]) + us + str(argv[4]));

  bwrite(x, pre + str(".bin"), nr[2], nc[2], nb[1] * 2);  // 2x as many floats for type-6
  hwrite(pre + str(".hdr"), nr[2], nc[2], nb[1], 6); // type 6
  
  // system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre + str(".bin 1 2 3 1")).c_str());
  return 0;
}
