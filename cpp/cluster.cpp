#include"misc.h"
size_t next_j, np, nrow, ncol, nband, kmax, k_use;
pthread_mutex_t next_j_mutex;
pthread_attr_t attr; // specify threads joinable

float * dat;
float * dmat_d; // = falloc(np * kmax);
unsigned int * dmat_i; // (unsigned int *)alloc(np * (size_t)kmax * (size_t)sizeof(unsigned int));
vector<unsigned int> top_i;

float * rho; // density estimate
unsigned int * label; // label assigned
unsigned int next_label;

void * dmat_threadfun(void * arg){
  float d, df;
  unsigned int i, ki;
  long k = (long)arg;
  unsigned int pi, u, my_i;
  long unsigned int my_next_j; //long k = (long)arg;
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

    if(my_next_j % 37 == 0){
      float pct = 100. * (float)next_j / (float) np;
      cprint(str(" worker: ") + to_string(k) + str(" pickup: ") + to_string(my_next_j) + str(" %") + to_string(pct));
    }

    priority_queue<f_idx> pq;
    for0(i, np){
      if(i != my_next_j){
        d = 0.;
        for(u = 0; u < nband; u++){
          pi = np * u;
          df = dat[pi + i] - dat[pi + my_next_j];
          d += df * df;
        }
        pq.push(f_idx(d, i)); // my_next_j
      }
    }

    for0(i, kmax){
      f_idx x(pq.top()); // dmat row is sorted
      ki = (my_next_j * kmax) + i;
      dmat_d[ki] = x.d;
      dmat_i[ki] = x.idx;
      pq.pop();
    }
    while(pq.size() != 0) pq.pop();
    my_i ++;
  }
}

unsigned int top(unsigned int j){
  unsigned int i, ki, ni;
  if(label[j] > 0){
    return label[j];
  }
  else{
    float rho_max = rho[j];
    unsigned int max_i = j;

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

int main(int argc, char** argv){
  kmax = 1111; // should probably modulate this somewhere. Probably good for practical purposes

  if(argc < 2) err("cluster [bin file name. hdr file must also be present");

  system("mkdir -p label");
  system("mkdir -p out");

  str bfn(argv[1]); // input "envi type-4" aka IEEE Floating-point 32bit BSQ (band sequential) data stack
  str hfn(hdr_fn(bfn)); // get name of header file
  hread(hfn, nrow, ncol, nband); // get image shape from header
  printf("nrow %d ncol %d nband %d\n", nrow, ncol, nband);
  dat = bread(bfn, nrow, ncol, nband); // read image data

  np = nrow * ncol;
  printf("np %d\n", np); // number of pixels
  printf("(np^2 - n) / 2=%f\n", (((float)np * (float)np) - (float)np) / 2.); // distance matrix size

  dmat_d = falloc(np * kmax); // scale data first? could put data scaling step here
  dmat_i = (unsigned int *)alloc(np * (size_t)kmax * (size_t)sizeof(unsigned int));

  if(fsize("dmat.d") < nrow * ncol * kmax * sizeof(float)){
    next_j = 0; // this variable is what the mutex (lock) goes on: "take a number"

    int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    printf("number of cores: %d\n", numCPU);

    pthread_mutex_init(&print_mtx, NULL); // mutex setup
    pthread_mutex_init(&next_j_mutex, NULL);
    pthread_attr_init(&attr); // make threads joinable
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t * my_pthread = new pthread_t[numCPU]; // allocate threads
    unsigned int j;
    for0(j, numCPU){
      long k = j;
      pthread_create(&my_pthread[j], &attr, dmat_threadfun, (void *) k);
    }

    for0(j, numCPU) pthread_join(my_pthread[j], NULL); // (re)calculate dmat, wait for threads to finish
    printf("end dmat_calc()\n");

    FILE * f = wopen("dmat.d"); // write distance from distance matrix
    fwrite(dmat_d, np * kmax * sizeof(float), 1, f);
    fclose(f);

    f = wopen("dmat.i"); // write index corresponding to dmat element in dmat.d
    if(!f) err("failed to open: dmat.i");
    fwrite(dmat_i, np * kmax * sizeof(unsigned int), 1, f);
    fclose(f);

  }
  else{
    FILE * f = fopen("dmat.d", "rb"); // restore dmat
    if(!f) err("failed to open dmat.d");
    fread(dmat_d, np * kmax * sizeof(float), 1, f);
    fclose(f);

    f = fopen("dmat.i", "rb");
    if(!f) err("failed to open dmat.i");
    fread(dmat_i, np * kmax * sizeof(unsigned int), 1, f);
    fclose(f);
  }

  rho = (float *)alloc(np * kmax * sizeof(float)); // buffer for density estimate
  label = (unsigned int *) alloc(np * sizeof(unsigned int)); // buffer for class label

  FILE * f = wopen("n_class.csv"); // record number of classes! Append to file on each run..
  fprintf(f, "n_classes,k_use"); // first record header line.. "k_use,n_classes");
  fclose(f);

  float d_avg;
  unsigned int i, j;

  // need to make arbitrary step, a parameter
  for(k_use = 1; k_use <= kmax; k_use += 10){

    top_i.clear();
    if(k_use > kmax) err("kuse > kmax"); //printf("density estimation..\n");

    for0(i, np){
      d_avg = 0.;
      for0(j, k_use) d_avg += dmat_d[i * kmax + j];
      rho[i] = 1. / d_avg;
    }

    next_label = 1; // start with label 1-- 0 for non-labelled / undefined
    top_i.push_back(0); // hill climbing; null is self-top
    for0(i, np) label[i] = 0; // default label: unlabeled
    for0(i, np) label[i] = top(i); // do the clustering
    printf("k_use %ld n_classes %ld\n", (long int)k_use, (long int)(next_label-1)); // print out K, # of classes

    f = fopen("n_class.csv", "ab"); // append number of classes
    if(!f) err("failed to open file: n_class.csv");
    fprintf(f, "\n%ld,%ld", (long int)(next_label-1), (long int)k_use); // (long int)(next_label-1));
    fclose(f);

    f = wopen(str("label/") + to_string(k_use) + str(".lab")); // write outputs
    float * label_float = falloc(np);
    for0(i, np) label_float[i] = (float)label[i];
    fwrite(label_float, np *sizeof(float), 1, f);
    free(label_float);
    fclose(f);

    f = wopen(str("out/") + to_string(k_use) + str(".bin"));
    int u;
    float df;
    unsigned int ti, pi;

    for0(u, nband){
      for0(i, np){
        pi = np * u;
        ti = top_i[label[i]];
        df = dat[pi + ti];
        fwrite(&df, sizeof(float), 1, f);
      }
    }
    fclose(f);
  }
  return 0;
}
