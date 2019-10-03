#include"misc.h"
size_t kmax, k_use;
pthread_attr_t attr; // specify threads joinable
pthread_mutex_t next_j_mutex;
size_t next_j;
size_t np;
int nrow, ncol, nband;
float * dat;

float * dmat_d;// = falloc(np * kmax);
unsigned int * dmat_i;// = (unsigned int *)alloc(np * (size_t)kmax * (size_t)sizeof(unsigned int));
vector<unsigned int> top_i;

float * rho;
unsigned int * label;
unsigned int next_label;

void * dmat_threadfun(void * arg){
  unsigned int i;
  long unsigned int my_next_j;
  long k = (long)arg;
  cprint(str("dmat_threadfun(") + std::to_string(k) + str(")"));
  float d, df;
  unsigned int pi;
  unsigned int my_i = 0;
  unsigned int u;

  while(1){
    // try to pick up a job
    pthread_mutex_lock(&next_j_mutex);
    my_next_j = next_j ++; // index of data this thread should pick up if it can
    pthread_mutex_unlock(&next_j_mutex);

    if(my_next_j > np -1){
      cprint(str("\texit thread ") + to_string(k));
      return(NULL);
    }

    if(my_next_j % 37 == 0){
      float pct = 100. * (float)next_j / (float) np;
      cprint(str(" worker: ") + to_string(k) + str(" pickup: ") + to_string(my_next_j)
      + str(" %") + to_string(pct));
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
        pq.push(f_idx(d, i)); //my_next_j));
      }
    }
    //dmat row is sorted
    for0(i, kmax){
      f_idx x(pq.top());
      unsigned int ki = (my_next_j * kmax) + i;
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
  if(label[j] > 0) return label[j];
  else{
    float rho_max = rho[j];
    unsigned int max_i = j;

    for0(i, k_use){
      ki = (j * kmax) + i;
      ni = dmat_i[ki];
      /*if(j == 0){
        printf("j %ld i %ld ki %ld ni %ld rho[ni] %f\n", j, i, ki, ni, rho[ni]);
        }*/
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
        top_i.push_back(j);
        // printf("next_label %ld np %ld\n", (long int)next_label, (long int)np);
        return label[j];
      }
    }
  }

  int main(int argc, char** argv){
    kmax = 2222;

    if(argc < 2) err("cluster [bin file name. hdr file must also be present");

    system("mkdir label");
    system("mkdir out");

    str bfn(argv[1]);
    //int nrow, ncol, nband;
    str hfn(split(bfn, '.')[0] + str(".hdr"));
    hread(hfn, nrow, ncol, nband);
    printf("nrow %d ncol %d nband %d\n", nrow, ncol, nband);

    dat = bread(bfn, nrow, ncol, nband);

    np = nrow * ncol;
    printf("np %d\n", np);
    printf("(np^2 - n) / 2=%f\n", (((float)np * (float)np) - (float)np) / 2.);

    // scale the data first?

    dmat_d = falloc(np * kmax);
    dmat_i = (unsigned int *)alloc(np * (size_t)kmax * (size_t)sizeof(unsigned int));

    if(fsize("dmat.d") < nrow * ncol * kmax * sizeof(float)){

      //----------------------------------------
      next_j = 0; // put a lock on this variable

      int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      cout << "Number of cores: " << numCPU << endl;

      // mutex setup
      pthread_mutex_init(&print_mutex, NULL);
      pthread_mutex_init(&next_j_mutex, NULL);

      // make the threads joinable
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

      // allocate threads
      pthread_t * my_pthread = new pthread_t[numCPU];
      unsigned int j;
      for0(j, numCPU){
        long k = j;
        pthread_create(&my_pthread[j], &attr, dmat_threadfun, (void *) k);
      }

      // wait for threads to finish
      for0(j, numCPU) pthread_join(my_pthread[j], NULL);
      cout << "end dmat_calc()" << endl;

      FILE * f;
      f = fopen("dmat.d", "wb");
      fwrite(dmat_d, np * kmax * sizeof(float), 1, f);
      fclose(f);
      f = fopen("dmat.i", "wb");
      fwrite(dmat_i, np * kmax * sizeof(unsigned int), 1, f);
      fclose(f);

    }
    else{
      FILE * f = fopen("dmat.d", "rb");
      fread(dmat_d, np * kmax * sizeof(float), 1, f);
      fclose(f);
      f = fopen("dmat.i", "rb");
      fread(dmat_i, np * kmax * sizeof(unsigned int), 1, f);
      fclose(f);
    }

    rho = (float *)alloc(np * kmax * sizeof(float));
    label = (unsigned int *) alloc(np * sizeof(unsigned int));

    FILE * f = fopen("nclass.csv", "wb");
    fprintf(f, "n_classes,k_use"); //"k_use,n_classes");
    fclose(f);

    for(k_use = 1; k_use <= kmax / 2; k_use++){
      top_i.clear();
      if(k_use > kmax) err("kuse > kmax");
      //printf("density estimation..\n");
      unsigned int i, j;
      for0(i, np){
        float d_avg = 0.;
        for0(j, k_use){
          d_avg += dmat_d[i * kmax + j];
        }
        rho[i] = 1. / d_avg;
      }

      //printf("hill climbing..\n");
      top_i.push_back(0); // null is self-top
      next_label = 1;
      for0(i, np) label[i] = 0; // default label: unlabeled

      // do the clustering
      for0(i, np) label[i] = top(i);

      printf("%ld %ld\n", (long int)k_use, (long int)(next_label-1));
      f = fopen("nclass.csv", "ab");
      fprintf(f, "\n%ld,%ld", (long int)(next_label-1), (long int)k_use);//, (long int)(next_label-1));
      fclose(f);

      // outputsa
      f = fopen((str("label/") + to_string(k_use) + str(".lab")).c_str(), "wb");
      float * label_float = falloc(np);
      for0(i, np) label_float[i] = (float)label[i];
      fwrite(label_float, np *sizeof(float), 1, f);
      free(label_float);
      fclose(f);

      f = fopen((str("out/") + to_string(k_use) + str(".out")).c_str(), "wb");
      int u;
      float df;
      for0(u, nband){
        for0(i, np){
          unsigned int ti = top_i[label[i]];
          unsigned int pi = np * u;
          df = dat[pi + ti];
          fwrite(&df, sizeof(float), 1, f);
        }
      }
      fclose(f);
    }
    return 0;
  }
