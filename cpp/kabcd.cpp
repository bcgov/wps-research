/* kabcd.cpp: 20221028 adapted from abcd.cpp to include K-nearest neighbours
instead of just nearest neighbour.a

Classification mode:
(*) object given most common class (among k-nearest neighbours) DONE.
Todo:
(*) expand this by outputting a tuple of the classes present, and the counts for each class. First most common, second most common, etc.
(*) develop an entropy concept to show if results are conflicted
(*) probability (pi) for each class, is the ratio of the count of observations (among the knn) for that class, divided by K (the number of neighbours)

Regression mode: (NOT implemented)
(*) Value for object is average of the values for the K-nearest neighbours. 
(*) add a standard deviation, min, max or other parameters to characterize the distribution and assess conflict.

 * 20220517 e.g.: abcd.exe A.bin B.bin C.bin # and compare w D.bin!
20220610 add skip_offset factor

How to project importance back on the dimensions?
Should windowing be added?

Also need change detection relative to a cyclic component / baseline
* Need missing values handling
*/
#include"misc.h"
static size_t nr[3], nc[3], nb[3], skip_f, skip_off, m, np, np2; // shapes
static float * y[3], *x, t, *A, *B, *C;  // data
static int * bp, * bp2;  // bad px: {A,B}, C respectively
static int knn_k; // number of neighbours to consider

void infer_px(size_t i){
  int debug = false;

  if(bp2[i]) return; // skip bad px in A, B
  priority_queue<f_i> pq;
  if(pq.size() != 0) err("nonempty pq");
  int m;
  float d, e; //, md = FLT_MAX;
  size_t j, k;
  size_t nb_0 = nb[0];
  size_t nb_1 = nb[1];

  for(j = skip_off; j < np; j += skip_f){  // uniform sample in space
    if(bp[j]) continue;  // skip bad px in C
    d = 0;
    for0(k, nb_0){
      e = A[np * k + j] - C[np2 * k + i];
      d += e * e;
    }
    vector<float> v;
    for0(k, nb_1) v.push_back(B[np * k + j]);  // check abcd.cpp
    if(debug){
      printf("push %e %zu ", d, j);
      cout << v << endl;
    }
    pq.push(f_i(d, j));
  }

  map<vector<float>, size_t> c;
  //map<vector<float>, size_t> exemplar;

  if(pq.size() < knn_k) err("not enough elements pushed");
  for0(k, knn_k){
    f_i t(pq.top());
    pq.pop();
    vector<float> v;
    for0(m, nb[1]) v.push_back(B[np * m + t.i]);
    
    if(debug){
      printf("%e %zu ", t.d, t.i);
      cout << v << endl;
    }

    if(c.count(v) < 1) c[v] = 1; // 1, 2, 3! 
    c[v] += 1;

    //exemplar[v] = t.i; // hold index of instance of vector
  }

  priority_queue<f_v> pq2;
  map<vector<float>, size_t>::iterator ci;
  for(ci = c.begin(); ci != c.end(); ci ++){
    if(debug) cout << "  " << ci->first << " " << ci->second << endl;
    pq2.push(f_v(ci->second, ci->first));
  }

  f_v top_val(pq2.top());
  while(pq2.size() > 0){
    f_v val(pq2.top());
    pq2.pop();
    if(debug) cout << "  " << val.d << " " << val.v << endl;
  }

  if(debug){
    cout << "mode: " << top_val.d << " " << top_val.v << endl;
    cout << "exit" << endl;
    exit(1);
  }

  // mi should be j of most frequent
  size_t mi = 0;
  for0(k, nb[1]){ // nb[2]) : need to revise in abcd.cpp?
    x[np2 * k + i] = top_val.v[k]; //B[np * k + mi];  // assign most frequent // not nearest
  }
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
  
  knn_k = 7;  // don't forget to set this at the command line later

  size_t i, n_bad;
  if(argc < 4){
    printf("A is to B as C is to ? Answer: D (the output)\n");
    printf("Note: A and B must have same shape (possibly different band count)\n");
    printf("Note: C's dimensions can differ (from A's and B's); C's band count matches A's\n");
    printf("The output result (i.e. \"D\") has the same dimensions as C\n");
    printf("D's band count wil match the band count of B\n"); 
    err("kabcd [A: img1 (n bands)] [B: img2 (m bands)] [C: img3 (n bands)] # [skip] # [skip_offset]\n");
  }
  skip_f = (argc > 4) ? (size_t) atol(argv[4]): 1; // bsq2bip -> binary_sort -> bip2bsq
  skip_off = (argc > 5) ? (size_t) atol(argv[5]): 0;

  for0(i, 3){
    hread(hdr_fn(argv[1 + i]), nr[i], nc[i], nb[i]);
  }

  // first pair of images have same dimensions, not necessarily same band count
  if(nr[0] != nr[1] || nc[0] != nc[1]){
    err("A.shape != B.shape");
  }

  // first and third images share band count
  if(nb[0] != nb[2]){
    err("A.n_bands != C.n_bands");
  }

  // first pair of images: equal dimensions (But not shape) hence pixel count
  (np = nr[0] * nc[0], np2 = nr[2] * nc[2]); // same with second pair!
  if(skip_f >= np){
    err("illegal skip_f");
  }

  // same shape as image C, should be same band count as image B
  // is this right? Do we need to fix this in abcd.cpp?
  x = falloc(nr[2] * nc[2] * nb[1]); // out buf (image "D")
  for0(i, nr[2] * nc[2] * nb[2]) x[i] = NAN;
  for0(i, 3) y[i] = bread(str(argv[i + 1]), nr[i], nc[i], nb[i]);  // read input

  (n_bad = 0, bp = ialloc(np));  // bad pixels in A, B?
  for0(i, np){
    bp[i] = is_bad(y[0], i, nb[0]) || is_bad(y[1], i, nb[1]);
    if(bp[i]) n_bad ++;
  }
  if(n_bad == np) err("no good pix: AxB");

  (n_bad = 0, bp2 = ialloc(np2));  // bad pxls in C?
  for0(i, np2){
    bp2[i] = is_bad(y[2], i, nb[2]);
    if(bp2[i]) n_bad ++;
  }
  if(n_bad == np2) err("no good pix: C");
  (A = y[0], B = y[1], C = y[2]);

  str u("_");
  parfor(0, np2, infer_px, 0); //1);  // inference by output pixel

  str pre(str("kabcd_") + str(argv[1]) + u + str(argv[2]) + u +
 	  		 str(argv[3]) + u + to_string(skip_f) + u +
			 		    to_string(skip_off));
  bwrite(x, pre + str(".bin"), nr[2], nc[2], nb[1]);  // write out
  hwrite(pre + str(".hdr"), nr[2], nc[2], nb[1]); // this info corroborates the choice above in alloc
  if(true){
    int r = system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre +
  	            str(".bin 1 2 3 1")).c_str());
  }
  run((str("envi_header_copy_bandnames.py ") + str(hdr_fn(argv[1 + 1])) + str(" ") + pre + str(".hdr")).c_str());
  //run(cmd);
  return 0;
}
