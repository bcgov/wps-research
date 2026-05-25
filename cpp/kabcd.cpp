/* kabcd.cpp: 20221028 adapted from abcd.cpp to include K-nearest neighbours
instead of just nearest neighbour.a

Classification mode:
(*) object given most common class (among k-nearest neighbours) DONE.
Todo:
(*) expand this by outputting a tuple of the classes present, and the counts for each class. First most common, second most common, etc.
(*) develop an entropy concept to show if results are conflicted
(*) probability (pi) for each class, is the ratio of the count of observations (among the knn) for that class, divided by K (the number of neighbours)

Regression mode:
(*) Value for object is average of the values for the K-nearest neighbours. DONE (--regress flag)
(*) add a standard deviation, min, max or other parameters to characterize the distribution and assess conflict.

 * 20220517 e.g.: abcd.exe A.bin B.bin C.bin # and compare w D.bin!
20220610 add skip_offset factor
20260505 add --regress flag for regression mode
20260525 fix is_bad to use correct pixel count for each image
20260525 add --validate: outlier detection via leave-one-out KNN distance
         and B-value std calibration on training data (regress mode only).
         Outputs 4 extra rasters: raw mean-dist, raw mean-std, flag-dist, flag-std.
         Thresholds at 95th and 99th percentile of training distribution.
         Disable with --no_validate when --regress is on.

How to project importance back on the dimensions?
Should windowing be added?

Also need change detection relative to a cyclic component / baseline
* Need missing values handling
*/
#include"misc.h"
#include<algorithm>
static size_t nr[3], nc[3], nb[3], skip_f, skip_off, m, np, np2; // shapes
static float * y[3], *x, t, *A, *B, *C;  // data
static int * bp, * bp2;  // bad px: {A,B}, C respectively
static int knn_k; // number of neighbours to consider
static bool regress_mode = false; // regression vs classification
static bool validate_mode = true; // outlier detection (regress only)

/* validation output buffers (np2 pixels each, 1 band) */
static float * val_dist;   // raw: mean KNN distance per pixel
static float * val_std;    // raw: mean B-value std per pixel
static float * flag_dist;  // flag: 0=ok, 1=marginal(p95), 2=outlier(p99)
static float * flag_std;   // flag: 0=ok, 1=marginal(p95), 2=outlier(p99)

/* validation thresholds (computed from training leave-one-out) */
static float thresh_dist_95, thresh_dist_99;
static float thresh_std_95, thresh_std_99;

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

  if(pq.size() < knn_k) err("not enough elements pushed");

  if(regress_mode){
    /* regression: average B-values over K nearest neighbours */
    vector<double> sum(nb_1, 0.);
    vector<double> sum2(nb_1, 0.); // for std computation
    double dist_sum = 0.;
    for0(k, knn_k){
      f_i t(pq.top());
      pq.pop();
      dist_sum += (double)t.d;
      for0(m, nb_1){
        double v = (double)B[np * m + t.i];
        sum[m] += v;
        sum2[m] += v * v;
      }
    }
    for0(k, nb_1) x[np2 * k + i] = (float)(sum[k] / (double)knn_k);

    if(validate_mode){
      /* mean KNN distance */
      float mean_d = (float)(dist_sum / (double)knn_k);
      val_dist[i] = mean_d;

      /* mean std of B-values across KNN, averaged over bands */
      double std_sum = 0.;
      for0(k, nb_1){
        double mu = sum[k] / (double)knn_k;
        double var = sum2[k] / (double)knn_k - mu * mu;
        if(var < 0.) var = 0.;
        std_sum += sqrt(var);
      }
      float mean_std = (float)(std_sum / (double)nb_1);
      val_std[i] = mean_std;

      /* flag using pre-computed thresholds */
      flag_dist[i] = (mean_d >= thresh_dist_99) ? 2.f :
                     (mean_d >= thresh_dist_95) ? 1.f : 0.f;
      flag_std[i]  = (mean_std >= thresh_std_99) ? 2.f :
                     (mean_std >= thresh_std_95) ? 1.f : 0.f;
    }
  }
  else{
    /* classification: mode of B-values over K nearest neighbours */
    map<vector<float>, size_t> c;

    for0(k, knn_k){
      f_i t(pq.top());
      pq.pop();
      vector<float> v;
      for0(m, nb[1]) v.push_back(B[np * m + t.i]);

      if(debug){
        printf("%e %zu ", t.d, t.i);
        cout << v << endl;
      }

      if(c.count(v) < 1) c[v] = 1;
      c[v] += 1;
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

    size_t mi = 0;
    for0(k, nb[1]){
      x[np2 * k + i] = top_val.v[k];
    }
  }
  if(i % 10000 == 0) status(i, np2);
}

inline int is_bad(float * dat, size_t i, size_t n_b, size_t n_pix){
  int zero = true;
  for0(m, n_b){  // find bad/empty pix
    t = dat[n_pix * m + i];
    if(isnan(t) || isinf(t)) return true;
    if(t != 0) zero = false;
  }
  return (n_b > 1 && zero); // 0 in 1-band product isn't bad
}

/* Leave-one-out KNN on training data to calibrate validation thresholds.
   Subsamples training pixels for efficiency, computes mean KNN distance
   and mean B-value std for each, then sets percentile thresholds. */
void calibrate_validation(){
  printf("calibrating validation thresholds (leave-one-out on training)...\n");

  /* collect indices of valid training pixels (respecting skip) */
  vector<size_t> good;
  for(size_t j = skip_off; j < np; j += skip_f){
    if(!bp[j]) good.push_back(j);
  }
  size_t n_good = good.size();
  printf("  valid training pixels (with skip): %zu\n", n_good);
  if(n_good < (size_t)(knn_k + 1)){
    err("not enough valid training pixels to calibrate");
  }

  /* subsample: use up to 10% or 50000, whichever is smaller */
  size_t max_sample = min((size_t)50000, n_good / 10);
  if(max_sample < (size_t)(knn_k + 1)) max_sample = min(n_good, (size_t)50000);
  size_t step = n_good / max_sample;
  if(step < 1) step = 1;

  vector<float> cal_dists;
  vector<float> cal_stds;
  size_t nb_0 = nb[0];
  size_t nb_1 = nb[1];

  for(size_t si = 0; si < n_good; si += step){
    size_t qi = good[si];  // query pixel index

    /* find KNN among training, skipping self */
    priority_queue<f_i> pq;
    for(size_t ti = 0; ti < n_good; ti++){
      size_t tj = good[ti];
      if(tj == qi) continue;
      float d = 0.f, e;
      for(size_t k = 0; k < nb_0; k++){
        e = A[np * k + qi] - A[np * k + tj];
        d += e * e;
      }
      pq.push(f_i(d, tj));
    }

    /* extract top-K */
    double dist_sum = 0.;
    vector<double> sum(nb_1, 0.);
    vector<double> sum2(nb_1, 0.);
    for(int ki = 0; ki < knn_k && pq.size() > 0; ki++){
      f_i top(pq.top());
      pq.pop();
      dist_sum += (double)top.d;
      for(size_t k = 0; k < nb_1; k++){
        double v = (double)B[np * k + top.i];
        sum[k] += v;
        sum2[k] += v * v;
      }
    }

    float mean_d = (float)(dist_sum / (double)knn_k);
    double std_sum = 0.;
    for(size_t k = 0; k < nb_1; k++){
      double mu = sum[k] / (double)knn_k;
      double var = sum2[k] / (double)knn_k - mu * mu;
      if(var < 0.) var = 0.;
      std_sum += sqrt(var);
    }
    float mean_std = (float)(std_sum / (double)nb_1);

    cal_dists.push_back(mean_d);
    cal_stds.push_back(mean_std);
  }

  /* sort and extract percentiles */
  size_t n_cal = cal_dists.size();
  printf("  calibration samples: %zu\n", n_cal);

  sort(cal_dists.begin(), cal_dists.end());
  sort(cal_stds.begin(), cal_stds.end());

  size_t i95 = (size_t)(0.95 * (n_cal - 1));
  size_t i99 = (size_t)(0.99 * (n_cal - 1));

  thresh_dist_95 = cal_dists[i95];
  thresh_dist_99 = cal_dists[i99];
  thresh_std_95  = cal_stds[i95];
  thresh_std_99  = cal_stds[i99];

  printf("  dist thresholds: p95=%.6e  p99=%.6e\n", thresh_dist_95, thresh_dist_99);
  printf("  std  thresholds: p95=%.6e  p99=%.6e\n", thresh_std_95, thresh_std_99);
}

int main(int argc, char** argv){
  
  knn_k = 7;

  /* scan for flags and remove them from argv */
  int new_argc = 0;
  char * new_argv[argc];
  for(int a = 0; a < argc; a++){
    if(str(argv[a]) == str("--regress")){
      regress_mode = true;
    }
    else if(str(argv[a]) == str("--knn_k") && a + 1 < argc){
      knn_k = atoi(argv[++a]);
      if(knn_k < 1) err("knn_k must be >= 1");
    }
    else if(str(argv[a]) == str("--no_validate")){
      validate_mode = false;
    }
    else{
      new_argv[new_argc++] = argv[a];
    }
  }
  argc = new_argc;
  argv = new_argv;

  /* validation only applies in regression mode */
  if(!regress_mode) validate_mode = false;

  cout << "Mode: " << (regress_mode ? "regression (average of KNN)" : "classification (mode of KNN)") << endl;
  cout << "K: " << knn_k << endl;
  cout << "Validation: " << (validate_mode ? "on" : "off") << endl;

  size_t i, n_bad;
  if(argc < 4){
    printf("A is to B as C is to ? Answer: D (the output)\n");
    printf("Note: A and B must have same shape (possibly different band count)\n");
    printf("Note: C's dimensions can differ (from A's and B's); C's band count matches A's\n");
    printf("The output result (i.e. \"D\") has the same dimensions as C\n");
    printf("D's band count wil match the band count of B\n");
    err("kabcd [A: img1 (n bands)] [B: img2 (m bands)] [C: img3 (n bands)] # [skip] # [skip_offset] [--regress] [--knn_k K] [--no_validate]\n");
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
  size_t a_s = nr[2] * nc[2] * nb[1];
  printf("alloc %zu\n", a_s);
  x = falloc(nr[2] * nc[2] * nb[1]); // out buf (image "D")
  for0(i, nr[2] * nc[2] * nb[1]) x[i] = NAN;
  for0(i, 3) y[i] = bread(str(argv[i + 1]), nr[i], nc[i], nb[i]);  // read input

  (n_bad = 0, bp = ialloc(np));  // bad pixels in A, B?
  for0(i, np){
    bp[i] = is_bad(y[0], i, nb[0], np) || is_bad(y[1], i, nb[1], np);
    if(bp[i]) n_bad ++;
  }
  if(n_bad == np) err("no good pix: AxB");

  (n_bad = 0, bp2 = ialloc(np2));  // bad pxls in C?
  for0(i, np2){
    bp2[i] = is_bad(y[2], i, nb[2], np2);
    if(bp2[i]) n_bad ++;
  }
  if(n_bad == np2) err("no good pix: C");
  (A = y[0], B = y[1], C = y[2]);

  /* allocate validation buffers and calibrate if needed */
  if(validate_mode){
    val_dist  = falloc(np2); for0(i, np2) val_dist[i]  = NAN;
    val_std   = falloc(np2); for0(i, np2) val_std[i]   = NAN;
    flag_dist = falloc(np2); for0(i, np2) flag_dist[i] = NAN;
    flag_std  = falloc(np2); for0(i, np2) flag_std[i]  = NAN;
    calibrate_validation();
  }

  str u("_");
  parfor(0, np2, infer_px, 0); //1);  // inference by output pixel

  str mode_str(regress_mode ? str("reg") : str("cls"));
  str pre(str("kabcd_") + mode_str + u + str(argv[1]) + u + str(argv[2]) + u +
                str(argv[3]) + u + to_string(skip_f) + u +
                         to_string(skip_off));
  bwrite(x, pre + str(".bin"), nr[2], nc[2], nb[1]);  // write out
  hwrite(pre + str(".hdr"), nr[2], nc[2], nb[1]);
  int r = system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre +
                  str(".bin 1 2 3 1")).c_str());
  run((str("envi_header_copy_bandnames.py ") + str(hdr_fn(argv[1 + 1])) + str(" ") + pre + str(".hdr")).c_str());
  run((str("envi_header_copy_mapinfo.py ") + str(hdr_fn(argv[1 + 1])) + str(" ") + pre + str(".hdr")).c_str());

  /* write validation rasters */
  if(validate_mode){
    str hdr_c(hdr_fn(argv[1 + 2]));  // copy mapinfo from C (same dims as output)

    str pre_vd(pre + str("_val_dist"));
    bwrite(val_dist,  pre_vd + str(".bin"), nr[2], nc[2], 1);
    hwrite(pre_vd + str(".hdr"), nr[2], nc[2], 1);
    run((str("envi_header_copy_mapinfo.py ") + hdr_c + str(" ") + pre_vd + str(".hdr")).c_str());
    r = system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre_vd +
                str(".bin 1 1 1 1")).c_str());

    str pre_vs(pre + str("_val_std"));
    bwrite(val_std,   pre_vs + str(".bin"), nr[2], nc[2], 1);
    hwrite(pre_vs + str(".hdr"), nr[2], nc[2], 1);
    run((str("envi_header_copy_mapinfo.py ") + hdr_c + str(" ") + pre_vs + str(".hdr")).c_str());
    r = system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre_vs +
                str(".bin 1 1 1 1")).c_str());

    str pre_fd(pre + str("_flag_dist"));
    bwrite(flag_dist, pre_fd + str(".bin"), nr[2], nc[2], 1);
    hwrite(pre_fd + str(".hdr"), nr[2], nc[2], 1);
    run((str("envi_header_copy_mapinfo.py ") + hdr_c + str(" ") + pre_fd + str(".hdr")).c_str());
    r = system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre_fd +
                str(".bin 1 1 1 1")).c_str());

    str pre_fs(pre + str("_flag_std"));
    bwrite(flag_std,  pre_fs + str(".bin"), nr[2], nc[2], 1);
    hwrite(pre_fs + str(".hdr"), nr[2], nc[2], 1);
    run((str("envi_header_copy_mapinfo.py ") + hdr_c + str(" ") + pre_fs + str(".hdr")).c_str());
    r = system((str("python3 ~/GitHub/wps-research/py/raster_plot.py ") + pre_fs +
                str(".bin 1 1 1 1")).c_str());

    printf("validation thresholds used:\n");
    printf("  dist: p95=%.6e  p99=%.6e\n", thresh_dist_95, thresh_dist_99);
    printf("  std:  p95=%.6e  p99=%.6e\n", thresh_std_95, thresh_std_99);
  }

  return 0;
}
