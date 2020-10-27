// MORE parallelization ....!!!!
// would be nice to have some dmat restore / parameter type features...
#include"misc.h"
size_t next_j, np, nrow, ncol, nband, kmax, k_use;
pthread_mutex_t next_j_mutex;
pthread_attr_t attr; // specify threads joinable

float * dat, * dmat_d; // = falloc(np * kmax);
size_t * dmat_i; // (unsigned int *)alloc(np * (size_t)kmax * (size_t)sizeof(unsigned int));
vector<size_t> top_i;

float * rho; // density estimate
size_t * label; // label assigned
size_t next_label;

void * dmat_threadfun(void * arg){
  // did we throw away the redundant half dmat we don't need?
  float d, df;
  size_t i, ki;
  size_t k = (size_t)arg;
  size_t pi, u, my_i;
  size_t my_next_j; //long k = (long)arg;
  cprint(str("dmat_threadfun(") + std::to_string(k) + str(")"));
  my_i = 0;

  while(1){
    pthread_mutex_lock(&next_j_mutex); // try to pick up a job
    my_next_j = next_j ++; // index of data this thread should pick up if it can
    pthread_mutex_unlock(&next_j_mutex);

    if(my_next_j > np -1){
      cprint(str("\texit thread ") + to_string(k));
      return(NULL);
    }

    if(my_next_j % 100 == 0){
      float pct = 100. * (float)next_j / (float) np;
      cprint(str(" worker: ") + to_string(k) + str(" job: ") + to_string(my_next_j) + str(" %") + to_string(pct));
    }

    priority_queue<f_idx> pq;
    for0(i, np){
      d = 0.;
      for(u = 0; u < nband; u++){
        pi = np * u;
        df = dat[pi + i] - dat[pi + my_next_j];
        d += df * df;
      }
      d = sqrt(d);
//      if(my_next_j == 0) printf("i %zu d %f\n", (size_t)i, d);
      
      pq.push(f_idx(d, i)); // my_next_j
    }

    for0(i, kmax){
      f_idx x(pq.top()); // dmat row already sorted
      ki = (my_next_j * kmax) + i;
      dmat_d[ki] = x.d;
      dmat_i[ki] = x.idx;
      pq.pop();
    }
    while(pq.size() != 0) pq.pop();
    my_i ++;
  }
}

size_t top(size_t j){
  size_t i, ki, ni;
  if(label[j] > 0){
    return label[j];
  }
  else{
    float rho_max = rho[j];
    size_t max_i = j;

    for0(i, k_use){
      ki = (j * kmax) + i;
      ni = dmat_i[ki]; // printf("j %ld i %ld ki %ld ni %ld rho[ni] %f\n", j, i, ki, ni, rho[ni]);

      if(rho[ni] > rho_max){
        rho_max = rho[ni];
        max_i = ni;
      }
    }
    if(max_i != j){
      return top(max_i);
    }
    else{
      label[j] = next_label ++;
      top_i.push_back(j); // printf("next_label %ld np %ld\n", (long int)next_label, (long int)np);
      return label[j];
    }
  }
}

void data_conditioning(float * dat, size_t nr, size_t nc, size_t nb){
  float d;
  size_t i, j, k;
  size_t np = nr * nc;

  for0(i, np){
    bool all_zero = true; // for each pixel
    for0(k, nb){
      d = dat[(k * np) + i]; // data value, this pixel, this band
      if(isnan(d) || isinf(d)){
        dat[(k * np) + i] = ((float)rand() / (float)RAND_MAX) / 111111111.; // replace NaN / inf with small random #s
      }
      else{
        if(d != 0.){
          all_zero = false;
        }
      }
    }
    if(all_zero){
      // if a pixel has all 0 values..
      for0(k, nb){
        dat[(k * np) + i] = ((float)rand() / (float)RAND_MAX) / 111111111.; // replace the values with small random #s
      }
    }
  }
}

void data_scaling(float * dat, size_t nr, size_t nc, size_t nb){
  float * min = falloc(nb);
  float * max = falloc(nb);
  size_t i, j, k, np;
  float d;

  for0(i, nb){
    min[i] = FLT_MAX;
    max[i] = FLT_MIN;
  }
  np = nr * nc;
  for0(k, nb){
    j = k * np;
    for0(i, np){
      d = dat[j + i];
      if(d < min[k]) min[k] = d;
      if(d > max[k]) max[k] = d;
    }
  }

  for0(k, nb) printf("K %zu min %f max %f\n", k, min[k], max[k]);

  // scale data to [0, 1]
  for0(k, nb){
    j = k * np;
    for0(i, np){
      dat[j + i] -= min[k];
      dat[j + i] /= (max[k] - min[k]);
    }
  }

  free(min);
  free(max);
}

int main(int argc, char** argv){
  srand(0);
  kmax = 1500; // should probably modulate this somewhere. Probably good for practical purposes

  if(argc < 2) err("cluster [bin file name. hdr file must also be present");

  system("mkdir -p label");
  system("mkdir -p out");
  system("mkdir -p mean");
  system("mkdir -p nearest");

  printf("%s\n", argv[1]);
  str bfn(argv[1]); // input "envi type-4" aka IEEE Floating-point 32bit BSQ (band sequential) data stack
  str hfn(hdr_fn(bfn)); // get name of header file
  hread(hfn, nrow, ncol, nband); // get image shape from header
  printf("nrow %d ncol %d nband %d\n", nrow, ncol, nband);
  dat = bread(bfn, nrow, ncol, nband); // read image data

  data_conditioning(dat, nrow, ncol, nband); // preconditioning on data
  data_scaling(dat, nrow, ncol, nband);

  np = nrow * ncol;
  printf("np %d\n", np); // number of pixels
  printf("(np^2 - n) / 2=%f\n", (((float)np * (float)np) - (float)np) / 2.); // distance matrix size

  dmat_d = falloc(np * kmax); // scale data first? could put data scaling step here
  dmat_i = (size_t *)alloc(np * (size_t)kmax * (size_t)sizeof(size_t));

  if(fsize("dmat.d") < nrow * ncol * kmax * sizeof(float)){
    next_j = 0; // this variable is what the mutex (lock) goes on: "take a number"

    size_t numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    printf("number of cores: %zu\n", numCPU);

    pthread_mutex_init(&print_mtx, NULL); // mutex setup
    pthread_mutex_init(&next_j_mutex, NULL);
    pthread_attr_init(&attr); // make threads joinable
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t * my_pthread = new pthread_t[numCPU]; // allocate threads
    size_t j;
    for0(j, numCPU){
      size_t k = j;
      pthread_create(&my_pthread[j], &attr, dmat_threadfun, (void *) k);
    }

    for0(j, numCPU) pthread_join(my_pthread[j], NULL); // (re)calculate dmat, wait for threads to finish
    printf("end dmat_calc()\n");

    FILE * f = wopen("dmat.d"); // write distance from distance matrix
    fwrite(dmat_d, np * kmax * sizeof(float), 1, f);
    fclose(f);

    f = wopen("dmat.i"); // write index corresponding to dmat element in dmat.d
    if(!f) err("failed to open: dmat.i");
    fwrite(dmat_i, np * kmax * sizeof(size_t), 1, f);
    fclose(f);

  }
  else{
    FILE * f = fopen("dmat.d", "rb"); // restore dmat
    if(!f) err("failed to open dmat.d");
    fread(dmat_d, np * kmax * sizeof(float), 1, f);
    fclose(f);

    f = fopen("dmat.i", "rb");
    if(!f) err("failed to open dmat.i");
    fread(dmat_i, np * kmax * sizeof(size_t), 1, f);
    fclose(f);
  }

  rho = (float *)alloc(np * kmax * sizeof(float)); // buffer for density estimate
  label = (size_t *) alloc(np * sizeof(size_t)); // buffer for class label

  FILE * f = wopen("n_class.csv"); // record number of classes! Append to file on each run..
  fprintf(f, "n_classes,k_use"); // first record header line.. "k_use,n_classes");
  fclose(f);

  float d_avg;
  size_t i, j, u;

  // need to make arbitrary step, a parameter???
  // also, can add parallelism here!!!!
  for(k_use = 1; k_use <= kmax; k_use += 10){

    top_i.clear();
    if(k_use > kmax) err("kuse > kmax"); //printf("density estimation..\n");

    for0(i, np){
      d_avg = 0.;
      for0(j, k_use){
        d_avg += dmat_d[i * kmax + j];
      }
      rho[i] = - d_avg;
      // rho[i] = 1. / d_avg;
    }

    next_label = 1; // start with label 1-- 0 for non-labelled / undefined
    top_i.push_back(0); // hill climbing; null is self-top
    for0(i, np) label[i] = 0; // default label: unlabeled
    for0(i, np) label[i] = top(i); // do the clustering
    size_t number_of_classes = next_label - 1;
    printf("k_use %zu n_classes %zu\n", k_use, number_of_classes); // print out K, # of classes

    f = fopen("n_class.csv", "ab"); // append number of classes
    if(!f) err("failed to open file: n_class.csv");
    fprintf(f, "\n%zu,%zu", number_of_classes, k_use); // (long int)(next_label-1));
    fclose(f);

    str lab_fn(str("label/") + zero_pad(to_string(k_use), 5));
    f = wopen(lab_fn + str(".bin")); // 1. write class outputs
    float * label_float = falloc(np);
    for0(i, np) label_float[i] = (float)label[i];
    fwrite(label_float, np *sizeof(float), 1, f);
    free(label_float);
    fclose(f);
    hwrite((lab_fn + str(".hdr")), nrow, ncol, 1);

    str out_fn(str("out/") + zero_pad(to_string(k_use), 5));
    f = wopen(out_fn + str(".bin")); // 2. write out hilltops
    float df;
    size_t ti, pi;

    for0(u, nband){
      for0(i, np){
        pi = np * u;
        ti = top_i[label[i]];
        df = dat[pi + ti];
        fwrite(&df, sizeof(float), 1, f);
      }
    }
    fclose(f);
    hwrite((out_fn + str(".hdr")), nrow, ncol, nband);

    str mean_fn(str("mean/") + zero_pad(to_string(k_use), 5)); // 3. write out means
    f = wopen(mean_fn + str(".bin"));
    float * nmean = falloc(number_of_classes); // variables to put means in
    float * means = falloc(number_of_classes * nband);
    for0(i, number_of_classes) nmean[i] = 0.; // start with 0.
    for0(i, (number_of_classes * nband)) means[i] = 0.; // start with 0.

    for0(i, np){
      u = label[i] - 1; // 0-indexed label for this pixel
      nmean[u] += 1.; // increment count for this 0-indexed label
      for0(j, nband){
        float dd = dat[(np * j) + i];
        means[(u * nband) + j] += dd; // for each data band (for this pixel)
      }
    }

    for0(i, number_of_classes){
      for0(j, nband){
        means[i * nband + j] /= nmean[i];
      }
    }

    for0(i, number_of_classes){
      for0(j, nband){
        float dd = means[(i * nband) + j];
        if(isinf(dd) || isnan(dd)){
          size_t idx = i * nband + j;
          printf("idx %zu\n", idx);
          printf("nclasses * nband %zu\n", number_of_classes * nband);
          err("stop");
        }
      }
    }

    for0(j, nband){
      for0(i, np){
        u = label[i] - 1; // for each band, for each pixel, get label
        df = means[(u * nband) + j];
        fwrite(&df, sizeof(float), 1, f);
      }
    }
    fclose(f);
    //free(nmean); // keep the means open!
    hwrite((mean_fn + str(".hdr")), nrow, ncol, nband);

    // so now that we have the means, do class re-assignment based on nearest mean

    str near_fn(str("nearest/") + zero_pad(to_string(k_use), 5)); // 3. write out labels
    for0(i, number_of_classes) nmean[i] = 0.;

    hwrite((near_fn + str(".hdr")), nrow, ncol, 1);
    float * nearest_mean = (float *) (void *) alloc(np * sizeof(float));
    for0(i, np){
      // if(i % 100 == 0) printf("i %zu\n", i);
      size_t k;
      size_t min_i = 0;
      float min_d = FLT_MAX;
      for0(k, number_of_classes){
        float d = 0.;
        for0(j, nband){
          float dd = dat[(np * j) + i] - means[(k * nband) + j];
          d += dd * dd;
        }
        d = sqrt(d);
        if(d < min_d){
          min_i = k;
          min_d = d;
        }
      }
      nearest_mean[i] = (float)min_i;
      nmean[min_i] += 1.;
    }
    bwrite(nearest_mean, (near_fn + str(".bin")), nrow, ncol, 1); // use this notation elsewhere?

    str nfnn(near_fn + str(".txt"));
    printf("*%s\n", nfnn.c_str());
    const char * nfn = nfnn.c_str();
    f = wopen(nfn);
    fprintf(f, "class_i,count");
    for0(i, number_of_classes){
      size_t class_i = (size_t)(i + 1);
      size_t class_count = (size_t)(nmean[i]);
      fprintf(f, "\n%zu,%zu", class_i, class_count);
    }
    fclose(f);

    free(nearest_mean);

    free(nmean);
    free(means);

    // OK what about re-assignment based on random samples from classes? as in kgc2010?
    //
    // ALSO PRINT OUT THE CLASS NUMBER FILES, FOR KGC RULE!!!!
  }
  return 0;
}
