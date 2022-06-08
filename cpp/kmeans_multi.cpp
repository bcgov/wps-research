#include"misc.h" // implementation of k-means algorithm 20201123
// added nan tolerance 20210120. Null class (0.) added which NAN pixels are assigned to

float tol; // tolerance percent
size_t nrow, ncol, nband, np, i, j, k, n, iter_max, K, nmf;// variables
float * dat, * means, * dmin, * dmax, *dcount, *mean, *label, *update;
bool * bad, * good;

void find_nearest(size_t i){
  size_t j, k; // find index of nearest cluster centre to this point
  size_t nearest_i = 0;
  float nearest_d = FLT_MAX;

  // don't assign bad points
  if(good[i]){
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
    nearest_i ++; // non-null label starts from 0
  }
  update[i] = nearest_i;
}

int main(int argc, char ** argv){
  if(argc < 3) err("kmeans [input binary file name] [k]");
  str fn(argv[1]); // input image file name
  K = atoi(argv[2]); // prescribed number of classes
  str hfn(hdr_fn(fn)); // input header file name
  hread(hfn, nrow, ncol, nband); // read header

  tol = 2.; // percent class change tolerance e.g. 2. is 2%!
  iter_max = 100.; // max number of iterations
  if(argc > 3) tol = atof(argv[3]);
  if(tol < 0 || tol >= 100) err("tol must be between 0 and 100.");

  np = nrow * ncol; // number of input pix
  dat = bread(fn, nrow, ncol, nband); // load floats to array
  means = falloc(np * nband); // output nearest mean for visualization
  bad = (bool *) alloc(sizeof(bool) * np);
  good = (bool *) alloc(sizeof(bool) * np);

  dmin = falloc(nband); // scale data between 0 and 1
  dmax = falloc(nband);
  for0(i, nband){
    dmin[i] = FLT_MAX;
    dmax[i] = FLT_MIN;
  }
  for0(i, np) bad[i] = false; // trust but verify

  for0(i, np){
    for0(k, nband){
      float d = dat[(np * k) + i];
      if(isnan(d) || isinf(d)){
        bad[i] = true; // mark this pixel as bad
      }
      else{
        if(d < dmin[k]) dmin[k] = d; // calculate min and max
        if(d > dmax[k]) dmax[k] = d;
      }
    }
  }
  size_t n_good = 0; // count good pixels
  for0(i, np){
    good[i] = !bad[i];
    if(good[i]) n_good ++;
  }

  for0(i, np){
    if(good[i]){
      for0(k, nband){
        // scale data
        dat[(np * k) + i] = (dat[(np * k) + i]- dmin[k]) / (dmax[k] - dmin[k]);
      }
    }
  }
  nmf = K * nband; // number of floats in means array
  dcount = falloc(K); // count
  mean = falloc(nmf); // mean vector for each class
  label = falloc(np); // init one label per pixel. 0 is non-labelled
  update = falloc(np); // new set of labels
  for0(i, np){
    if(good[i]){
      label[i] = i % K; // uniform initialization outside of null label
    }
    else{
      label[i] = NAN; // null / nonclass label
    }
  }

  for0(n, iter_max){
    // still more to parallelize here!
    for0(k, K) dcount[k] = 0.; // denominator for average
    for0(i, nmf) mean[i] = 0.; // for each iter, calculate class means
    for0(i, np){
      if(good[i] && label[i] > 0. && label[i] < (float)K){
        for0(k, nband){
		mean[((size_t)label[i] * nband) + k] += dat[(np * k) + i];
	}
        dcount[(size_t)label[i]] += 1;
      }
    }
    for0(i, K) if(dcount[i] > 0) for0(j, nband) mean[(i * nband) + j] /= dcount[i]; // mean = total / count

    for0(i, K){
      printf("K=%zu ", i);
      for0(j, nband) printf("%f ", mean[(i * nband) + j]);
      printf("\n");
    }
    parfor(0, np, find_nearest); // find nearest centre to each point

    size_t n_change = 0;
    for0(i, np){
      if(good[i]){
        if(label[i] != update[i]) n_change ++;
      }
    }
    float pct_chg = 100. * (float)n_change / (float)n_good; // plot change info
    printf("iter %zu of %zu n_change %f\n", n + 1, iter_max, pct_chg);
    set<size_t> observed; for0(i, np) observed.insert(label[i]); cout << " " << observed << endl; // plot observed labels */
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
  for0(i, np){
    if(good[i]){
      for0(k, nband){
        means[(k * np) + i] = mean[((size_t)label[i] * nband) + k]; // colour by mean
      }
    }
  }
  bwrite(means, omn, nrow, ncol, nband);
  hwrite(omh, nrow, ncol, nband, 4); // write data

  free(bad);
  return 0;
}
