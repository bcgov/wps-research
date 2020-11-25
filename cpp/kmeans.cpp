//implementation of k-means algorithm, with envi type-4 image input. Use patch?
#include"misc.h"
int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, i, j, k, n;// variables
  if(argc < 3) err("kmeans [input binary file name] [k]");
  str fn(argv[1]); // input image file name
  str hfn(hdr_fn(fn)); // input header file name
  size_t iter_max = 100;
  size_t K = atoi(argv[2]); // prescribed number of classes
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * means = falloc(np * nband); // output nearest mean for visualization

  // scale data between 0 and 1
  float * min = falloc(nband);
  float * max = falloc(nband);
  for0(i, nband){
    min[i] = FLT_MAX;
    max[i] = FLT_MIN;
  }

  for0(i, np){
    for0(k, nband){
      float d = dat[(np * k) + i];
      if(d < min[k]) min[k] = d;
      if(d > max[k]) max[k] = d;
    }
  }

  for0(i, np){
    for0(k, nband){
      float d = dat[(np * k) + i];
      dat[(np * k) + i] = (d - min[k]) / (max[k] - min[k]);
    }
  }

  float * label = falloc(np); // init one label per pixel. 0 is non-labelled
  float * update = falloc(np); // new set of labels
  for0(i, np) label[i] = (i % K); // uniform initialization
  size_t nmf = K * nband; // number of floats in means array
  float * mean = falloc(nmf); // mean vector for each class
  float * count = falloc(K); // count

  for0(n, iter_max){
    for0(i, nmf) mean[i] = 0.; // for each iter, calculate class means
    for0(k, K) count[k] = 0.; // denominator for average
    for0(i, np){
      for0(k, nband){
        float d = dat[(np * k) + i];
        mean[(((size_t)label[i]) * nband) + k] += d;
      }
      count[(size_t)label[i]] += 1;
    }
    for0(i, K) if(count[i] > 0) for0(j, nband) mean[(i * nband) + j] /= count[i];

    // for each point, reassign to nearest cluster centre
    for0(i, np){
      size_t nearest_i = 0;
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

    size_t n_change = 0;
    for0(i, np) if(label[i] != update[i]) n_change ++;
    float pct_chg = 100. * (float)n_change / (float)np;
    printf("iter %zu of %zu n_change %f ", n + 1, iter_max, pct_chg);

    set<size_t> observed; // plot observed labels
    for0(i, np) observed.insert(label[i]);
    cout << " " << observed << endl;

    float * tmp = label; // swap op
    label = update;
    update = tmp;

    for0(i, np) update[i] = 0.;
    if(pct_chg < 1.) break; // close enough? stop iterating if <1% of pixels changed class
  }

  str ofn(str(argv[1]) + str("_kmeans.bin")); // output class labels
  str ohn(str(argv[1]) + str("_kmeans.hdr"));
  bwrite(label, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1, 4); // write type 4 header

  str omn(str(argv[1]) + str("_means.bin")); // output class centres
  str omh(str(argv[1]) + str("_means.hdr"));
  for0(i, np){
    size_t my_label = label[i];
    for0(k, nband){
      means[(k * np) + i] = mean[(my_label * nband) + k];
    }
  }
  bwrite(means, omn, nrow, ncol, nband);
  hwrite(omh, nrow, ncol, nband, 4);
  return 0;
}