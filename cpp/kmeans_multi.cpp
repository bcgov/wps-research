#include"misc.h" // implementation of k-means algorithm 20201123

float tol; // tolerance percent
size_t nrow, ncol, nband, np, i, j, k, n, iter_max, K, nmf;// variables
float * dat, * means, * dmin, * dmax, *dcount, *mean, *label, *update;

void find_nearest(size_t i){
  size_t j, k;
  size_t nearest_i = 0; // for each point, reassign to nearest cluster centre
  float nearest_d = FLT_MAX;

  for0(j, K){
    float dd = 0.; // distance from this point to centre
    for0(k, nband){
      float d = dat[(np * k) + i] - mean[(j * nband) + k];
      dd += d * d;
    }
    dd = sqrt(dd);
    if(dd < nearest_d){
      nearest_d = dd;
      nearest_i = j;
    }
  }
  update[i] = nearest_i;
}

int main(int argc, char ** argv){
  if(argc < 3) err("kmeans [input binary file name] [k]");

  iter_max = 100;
  str fn(argv[1]); // input image file name
  K = atoi(argv[2]); // prescribed number of classes
  str hfn(hdr_fn(fn)); // input header file name
  hread(hfn, nrow, ncol, nband); // read header
  tol = 1.;
  if(argc > 3) tol = atof(argv[3]);
  if(tol < 0 || tol >= 100) err("tol must be 0 and 100.");



  np = nrow * ncol; // number of input pix
  dat = bread(fn, nrow, ncol, nband); // load floats to array
  means = falloc(np * nband); // output nearest mean for visualization

  dmin = falloc(nband); // scale data between 0 and 1
  dmax = falloc(nband);
  for0(i, nband){
    dmin[i] = FLT_MAX;
    dmax[i] = FLT_MIN;
  }

  for0(i, np){
    for0(k, nband){
      float d = dat[(np * k) + i];
      if(d < dmin[k]) dmin[k] = d;
      if(d > dmax[k]) dmax[k] = d;
    }
  }
  for0(i, np) for0(k, nband) dat[(np * k) + i] = (dat[(np * k) + i]- dmin[k]) / (dmax[k] - dmin[k]); // scale data

  nmf = K * nband; // number of floats in means array
  dcount = falloc(K); // count
  mean = falloc(nmf); // mean vector for each class
  label = falloc(np); // init one label per pixel. 0 is non-labelled
  update = falloc(np); // new set of labels
  for0(i, np) label[i] = (i % K); // uniform initialization

  for0(n, iter_max){
    // still more to parallelize here!
    for0(k, K) dcount[k] = 0.; // denominator for average
    for0(i, nmf) mean[i] = 0.; // for each iter, calculate class means
    for0(i, np){
      for0(k, nband) mean[(((size_t)label[i]) * nband) + k] += dat[(np * k) + i];
      dcount[(size_t)label[i]] += 1;
    }
    for0(i, K) if(dcount[i] > 0) for0(j, nband) mean[(i * nband) + j] /= dcount[i];

    parfor(0, np, find_nearest);

    size_t n_change = 0;
    for0(i, np) if(label[i] != update[i]) n_change ++;
    float pct_chg = 100. * (float)n_change / (float)np; // plot change info
    printf("iter %zu of %zu n_change %f\n", n + 1, iter_max, pct_chg);
    /*
    set<size_t> observed; // plot observed labels
    for0(i, np) observed.insert(label[i]);
    cout << " " << observed << endl;
    */
    float * tmp = label; // swap
    label = update;
    update = tmp;

    for0(i, np) update[i] = 0.; // close enough? stop iterating if <1% of pix changed class
    if(pct_chg < tol) break;
  }

  str ofn(str(argv[1]) + str("_kmeans.bin")); // output class labels
  str ohn(str(argv[1]) + str("_kmeans.hdr"));
  hwrite(ohn, nrow, ncol, 1, 4); // write type 4 header
  bwrite(label, ofn, nrow, ncol, 1); // write data

  str omn(str(argv[1]) + str("_means.bin")); // output class centres, for each pixel categorized
  str omh(str(argv[1]) + str("_means.hdr"));
  for0(i, np) for0(k, nband) means[(k * np) + i] = mean[((size_t)label[i] * nband) + k]; // colour by mean
  bwrite(means, omn, nrow, ncol, nband);
  hwrite(omh, nrow, ncol, nband, 4); // write data
  return 0;
}
