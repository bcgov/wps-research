/* 20220517 vim view_as.cpp; clean; python3 compile.py; ./view_as.exe A.bin B.bin C.bin # compare with D.bin
vim view_as.cpp; clean; python3 compile.py; valgrind ./view_as.exe A.bin B.bin C.bin # add g flag for valgrind
Also:
view_as.exe A.bin B.bin A.bin
view_as.exe A.bin A.bin A.bin

Easy: bin the input data? Or both?

NEED TO FIX BUG:
LAST IMAGE DIMENSION DOES NOT NEED TO MATCH! */
#include"misc.h"

static size_t nr[3], nc[3], nb[3], skip_f;
static float * y[3], *x;
static size_t np, np2;

void job(size_t i){
  float d;
  size_t j, k;
  size_t mi = 0;
  bool bad = false;
  bool zero = true;
  float md = FLT_MAX;

  for0(k, nb[0]){
    d = y[0][np * k + i];
    if(isnan(d) || isinf(d)) bad = true;
    if(d != 0) zero = false;
  }

  if(zero) bad = true;
  if(bad) return;

  for(j = 0; j < np; j += skip_f){
    bad = false;
    zero = true;

    for0(k, nb[0]){
      d = y[2][np * k + j];
      if(isnan(d) || isinf(d)) bad = true;
      if(d != 0) zero = false;
    }
    if(zero) bad = true;
    if(bad) continue;

    d = 0;
    for0(k, nb[0])
      d += (y[0][np * k + i] - y[2][np * k + j]) * (y[0][np * k + i] - y[2][np * k + j]);

    if(d < md){
      md = d;
      mi = j;
    }
  }

  if(i % 100000 == 0)
    cprint(to_string(100.* ((float)(i+1) / (float)np)) + str(" % ") + to_string(i) + str(" / ") + to_string(np));

  for0(k, nb[2])
    x[np2 * k + i] = y[1][np2 * k + mi];
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

  parfor(0, np2, job);
  str pre(str("view_as_") +
	  str(argv[1]) + str("_") + str(argv[2]) + str("_") +
	  str(argv[3]) + str("_") + str(argv[4]));

  bwrite(x, pre + str(".bin"), nr[0], nc[0], nb[1]);
  hwrite(pre + str(".hdr"), nr[0], nc[0], nb[1]);

  system((str("python3 ~/GitHub/bcws-psu-research/py/raster_plot.py ") +
	 pre + str(".bin 1 2 3 1")).c_str());
  return 0;
}
