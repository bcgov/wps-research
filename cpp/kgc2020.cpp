/* kgc2020. 20201101
Before adding new gradient rule. Verified
preconditioning and setoid dedup */
#include"misc.h"
size_t * dmat_i;
pthread_attr_t attr; // specify threads joinable
pthread_mutex_t nxt_j_mtx;
size_t nxt_j, np, nr, nc, nb, kmax, k_use;
float * dat, * dmat_d;

vector<size_t> top_i;
float * rho; // dens. est.
size_t * label;
size_t next_label; // label assigned, next label

size_t skip_factor = 1; // NEW: skip factor for sampling
size_t n_sampled = 0;   // NEW: number of sampled points
size_t * sample_idx;    // NEW: indices of sampled points in deduplicated set
float * sampled_dat;    // NEW: data for sampled points only
float * label_out;      // NEW: output labels for all pixels

void * dmat_threadfun(void * arg){
  float d, df; // do we throw away redundant half of dmat?
  size_t i, ki, pi, u, my_i, my_nxt_j, k;
  k = (size_t)arg;
  my_i = 0;
  cprint(str("dmat_threadfun(") + std::to_string(k) + str(")"));
 
  priority_queue<f_idx> pq;
  while(1){
    pthread_mutex_lock(&nxt_j_mtx); // try to pick up a job
    my_nxt_j = nxt_j ++; // index of data this thread should pick up if it can
    pthread_mutex_unlock(&nxt_j_mtx);

    if(my_nxt_j > np -1){
      cprint(str("\texit thread ") + to_string(k));
      return(NULL);
    }
    if(my_nxt_j % 100 == 0){
      float pct = 100. * (float)nxt_j / (float) np;
      cprint(str(" worker: ") + to_string(k) + str(" job: ") + to_string(my_nxt_j) + str(" %") + to_string(pct));
    }

    size_t ji = my_nxt_j * nb;
    for0(i, np){
      d = 0.;
      pi = i * nb;
      for(u = 0; u < nb; u++){
        df = dat[pi + u] - dat[ji + u];
        d += df * df;
      }
      d = sqrt(d);
      pq.push(f_idx(d, i)); // my_nxt_j
    }

    for0(i, kmax){
      f_idx x(pq.top()); // dmat row already sorted
      ki = (my_nxt_j * kmax) + i;
      dmat_d[ki] = x.d;
      dmat_i[ki] = x.idx;
      pq.pop();
    }
    while(pq.size() != 0) pq.pop();
    my_i ++;
  }
}

// NEW: thread function to assign labels to non-sampled deduplicated points
void * assign_label_threadfun(void * arg){
  float d, df, d_min;
  size_t i, u, my_nxt_j, k, nearest_idx;
  k = (size_t)arg;
  cprint(str("assign_label_threadfun(") + std::to_string(k) + str(")"));

  while(1){
    pthread_mutex_lock(&nxt_j_mtx);
    my_nxt_j = nxt_j++;
    pthread_mutex_unlock(&nxt_j_mtx);

    if(my_nxt_j >= np){
      cprint(str("\texit assign thread ") + to_string(k));
      return(NULL);
    }
    if(my_nxt_j % 1000 == 0){
      float pct = 100. * (float)my_nxt_j / (float)np;
      cprint(str(" assign worker: ") + to_string(k) + str(" job: ") + to_string(my_nxt_j) + str(" %") + to_string(pct));
    }

    // find nearest sampled point
    d_min = FLT_MAX;
    nearest_idx = 0;
    size_t ji = my_nxt_j * nb;
    for0(i, n_sampled){
      d = 0.;
      size_t pi = i * nb;
      for(u = 0; u < nb; u++){
        df = dat[ji + u] - sampled_dat[pi + u];
        d += df * df;
      }
      if(d < d_min){
        d_min = d;
        nearest_idx = i;
      }
    }
    label[my_nxt_j] = label[sample_idx[nearest_idx]];
  }
}

// NEW: write mean image where each pixel gets the vector value of its class exemplar
void write_mean_image(size_t k_use, size_t nr, size_t nc, size_t nb, 
                      size_t * ddup_lookup, float * dat_full, size_t n_ddup){
  size_t i, k, np_img = nr * nc;
  
  str mean_fn(str("mean/") + zero_pad(to_string(k_use), 5));
  FILE * f = wopen(mean_fn + str(".bin"));
  
  float * mean_out = falloc(np_img * nb);
  
  for0(i, np_img){
    size_t ddup_idx = ddup_lookup[i];         // pixel -> deduplicated index
    size_t lab = label[ddup_idx];             // deduplicated index -> class label
    size_t top_idx = top_i[lab];              // class label -> exemplar index (in sampled set)
    size_t exemplar_ddup;
    if(skip_factor > 1){
      exemplar_ddup = sample_idx[top_idx];    // sampled index -> deduplicated index
    } else {
      exemplar_ddup = top_idx;                // skip_factor==1: sampled index equals deduplicated index
    }
    
    for0(k, nb){
      mean_out[i * nb + k] = dat_full[exemplar_ddup * nb + k];
    }
  }
  
  fwrite(mean_out, np_img * nb * sizeof(float), 1, f);
  free(mean_out);
  fclose(f);
  
  hwrite((mean_fn + str(".hdr")), nr, nc, nb);
}

size_t top(size_t j){
  size_t i, ki, ni;
  if(label[j] > 0) return label[j];

  else{
    float rho_max = rho[j];
    size_t max_i = j;

    for0(i, k_use){
      ki = (j * kmax) + i;
      ni = dmat_i[ki];
      if(rho[ni] > rho_max){
        rho_max = rho[ni];
        max_i = ni;
      }
    }

    if(max_i != j) return top(max_i);
    else{
      label[j] = next_label ++;
      top_i.push_back(j);
      return label[j];
    }
  }
}

int main(int argc, char ** argv){
  kmax = 2000;
  cout << "dmat.exe" << endl; // shuffle data according to deduplication index
  if(argc < 3) err("dmat.exe [input file bip format] [deduplication index file _dedup] [--nskip skip_factor]");

  // NEW: parse --nskip argument
  for(int a = 3; a < argc; a++){
    if(strcmp(argv[a], "--nskip") == 0 && a + 1 < argc){
      skip_factor = (size_t)atol(argv[a + 1]);
      if(skip_factor < 1) skip_factor = 1;
      printf("skip_factor: %zu\n", skip_factor);
      a++;
    }
  }

  float d;
  size_t i, j, k, m, n;
  str inf(argv[1]); // input file
  str hfn(hdr_fn(inf)); // input header
  hread(hfn, nr, nc, nb); // read input hdr
  np = nr * nc; // number of pixels
  dat = bread(inf, nr, nc, nb); // read bip data

  str dpf(argv[2]);
  if(!exists(dpf)) err("failed to open deduplicated data index file");
  size_t ddup_s = fsize(dpf);
  size_t n_ddup = ddup_s / sizeof(size_t); // deduplicated data count
  printf("deduplicated data count: %zu\n", n_ddup);

  size_t * ddup_i = (size_t *)alloc(n_ddup * sizeof(size_t));
  FILE * f = ropen(dpf);
  size_t n_r = fread(ddup_i, sizeof(size_t), n_ddup, f);
  if(n_r != n_ddup) err("unexpected record count read");
  fclose(f);

  float * dat_0 = bread(inf, nr, nc, nb); // read the original dataset
  dat = falloc(nb * n_ddup); // allocate floats for deduplicated set
  for0(i, n_ddup * nb) dat[i] = 0.;

  m = n = 0;
  for0(i, n_ddup){
    n = nb * ddup_i[i]; // location of deduplicated record
    for0(k, nb) dat[m++] = dat_0[n++];
  }

  float * d_min= falloc(nb);  // scale to [0, 1]
  float * d_max = falloc(nb);
  for0(i, nb){
    d_min[i] = FLT_MAX;
    d_max[i] = -FLT_MAX;
  }

  for0(i, n_ddup){
    m = i * nb;
    for0(k, nb){
      d = dat[m + k];
      if(d < d_min[k]) d_min[k] = d;
      if(d > d_max[k]) d_max[k] = d;
    }
  }

  for0(i, n_ddup){
    m = i * nb;
    for0(k, nb){
      dat[m + k] -= d_min[k];
      dat[m + k] /= (d_max[k] - d_min[k]);
    }
  }

  printf("min ");
  for0(i, nb) printf(" %f", d_min[i]);
  printf("\nmax ");
  for0(i, nb) printf(" %f", d_max[i]);
  printf("\n");

  str of2(inf + str("_dedup_lookup"));
  size_t * ddup_lookup = (size_t *) alloc(sizeof(size_t) * np);
  f = ropen(of2);
  n_r = fread(ddup_lookup, sizeof(size_t), np, f);
  fclose(f);
  if(n_r != np) err("unexpected record read count");
  if(np != nr * nc) err("unexpected record read count");

  // NEW: compute sampled indices
  n_sampled = (n_ddup + skip_factor - 1) / skip_factor;
  sample_idx = (size_t *)alloc(n_sampled * sizeof(size_t));
  sampled_dat = falloc(n_sampled * nb);
  
  size_t si = 0;
  for(i = 0; i < n_ddup && si < n_sampled; i += skip_factor){
    sample_idx[si] = i;
    for0(k, nb) sampled_dat[si * nb + k] = dat[i * nb + k];
    si++;
  }
  n_sampled = si;
  printf("sampled data count: %zu (skip_factor=%zu)\n", n_sampled, skip_factor);

  np = n_sampled; // NEW: use sampled count for dmat computation
  kmax = kmax > np? np : kmax; // make sure kmax isn't more than the number of points we have!
  printf("kmax %zu\n", kmax);

  printf("np %d\n", np); // number of pixels
  printf("(np^2 - n) / 2=%f\n", (((float)np * (float)np) - (float)np) / 2.); // distance matrix size
  dmat_d = falloc(np * kmax); // scale data first? could put data scaling step here
  dmat_i = (size_t *)alloc(np * (size_t)kmax * (size_t)sizeof(size_t));

  // NEW: use sampled data for dmat computation
  float * dat_backup = dat;
  dat = sampled_dat;

  str dmatd_fn(inf + str("_") + to_string(kmax) + str("_skip") + to_string(skip_factor) + str("_dmat.d"));
  str dmati_fn(inf + str("_") + to_string(kmax) + str("_skip") + to_string(skip_factor) + str("_dmat.i"));

  if(fsize(dmatd_fn) != np * kmax * sizeof(float)){
    nxt_j = 0; // this variable is what the mutex (lock) goes on: "take a number"

    size_t numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    printf("number of cores: %zu\n", numCPU);

    pthread_mutex_init(&print_mtx, NULL); // mutex setup
    pthread_mutex_init(&nxt_j_mtx, NULL);
    pthread_attr_init(&attr); // make threads joinable
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t * my_pthread = new pthread_t[numCPU]; // allocate threads
    for0(j, numCPU) pthread_create(&my_pthread[j], &attr, dmat_threadfun, (void *) j);

    for0(j, numCPU) pthread_join(my_pthread[j], NULL); // (re)calculate dmat, wait for threads to finish
    printf("end dmat_calc()\n");

    FILE * f = wopen(dmatd_fn); // write distance from distance matrix
    fwrite(dmat_d, np * kmax * sizeof(float), 1, f);
    fclose(f);

    f = wopen(dmati_fn); // write index corresponding to dmat element in dmat.d
    fwrite(dmat_i, np * kmax * sizeof(size_t), 1, f);
    fclose(f);

  }
  else{
    FILE * f = ropen(dmatd_fn); // restore dmat
    fread(dmat_d, np * kmax * sizeof(float), 1, f);
    fclose(f);

    f = ropen(dmati_fn);
    fread(dmat_i, np * kmax * sizeof(size_t), 1, f);
    fclose(f);
  }

  rho = (float *)alloc(np * kmax * sizeof(float)); // buffer for density estimate
  label = (size_t *) alloc(n_ddup * sizeof(size_t)); // NEW: allocate for all deduplicated points

  f = wopen("n_class.csv"); // record number of classes! Append to file on each run..
  fprintf(f, "n_classes,k_use"); // first record header line.. "k_use,n_classes");
  fclose(f);

  long int last_number_of_classes = -1;
  for(k_use = 1; k_use <= kmax; k_use += 15){
    np = n_sampled; // NEW: use sampled count
    top_i.clear();
    if(k_use > kmax) err("kuse > kmax"); //printf("density estimation..\n");

    for0(i, np){
      float d_avg = 0.;
      for0(j, k_use){
        d_avg += dmat_d[i * kmax + j];
      }
      rho[i] = - d_avg;  //rho[i] = 1. / d_avg;
    }

    next_label = 1; // start with label 1-- 0 for non-labelled / undefined
    top_i.push_back(0); // hill climbing; null is self-top
    for0(i, n_ddup) label[i] = 0; // NEW: clear all labels
    for0(i, np) label[sample_idx[i]] = top(i); // NEW: cluster sampled points, store at original indices
    size_t number_of_classes = next_label - 1;
    printf("k_use %zu n_classes %zu\n", k_use, number_of_classes); // print out K, # of classes

    // NEW: assign labels to non-sampled deduplicated points using pthread parallelism
    if(skip_factor > 1){
      printf("assigning labels to non-sampled points...\n");
      dat = dat_backup; // restore full deduplicated data
      nxt_j = 0;
      np = n_ddup;

      size_t numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      pthread_t * assign_pthread = new pthread_t[numCPU];
      for0(j, numCPU) pthread_create(&assign_pthread[j], &attr, assign_label_threadfun, (void *)j);
      for0(j, numCPU) pthread_join(assign_pthread[j], NULL);
      delete[] assign_pthread;
      printf("label assignment complete\n");
      
      dat = sampled_dat; // restore for next iteration
    }
   
    f = fopen("n_class.csv", "ab"); // append number of classes
    if(!f) err("failed to open file: n_class.csv");
    fprintf(f, "\n%zu,%zu", number_of_classes, k_use); // (long int)(next_label-1));
    fclose(f); 
 
    // check output folders
    system("mkdir -p label");  // check output folders
    system("mkdir -p nearest");
    system("mkdir -p mean");
    system("mkdir -p out");

    // write label binary
    np = nr * nc; // not n_ddup anymore! map back onto pixels
    
    if(number_of_classes != last_number_of_classes){
      str lab_fn(str("label/") + zero_pad(to_string(k_use), 5));
      f = wopen(lab_fn + str(".bin")); // 1. write class outputs
      float * label_float = falloc(np);
      for0(i, np) label_float[i] = (float)label[ddup_lookup[i]];
      fwrite(label_float, np *sizeof(float), 1, f);
      free(label_float);
      fclose(f);
      hwrite((lab_fn + str(".hdr")), nr, nc, 1);

      // NEW: write mean image
      write_mean_image(k_use, nr, nc, nb, ddup_lookup, dat_backup, n_ddup);
    }

    if(number_of_classes == 1) break;
    last_number_of_classes = number_of_classes;
  }

  free(ddup_i);
  free(dat_0);
  free(dat_backup);
  free(sample_idx);
  free(sampled_dat);
  return 0;
}


