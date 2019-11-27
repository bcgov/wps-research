/*
dent cpp/hclust.cpp; vim cpp/hclust.cpp; rm bin/hclust; python setup.py; ./bin/hclust data/fran/mS2.bin data/fran/mS2.bin_label.bin 10
dent cpp/hclust.cpp; vim cpp/hclust.cpp; rm bin/hclust; python setup.py; ./bin/hclust data/fran/mS2.bin 10
*/

#include"misc.h"

size_t d_count;
float * dat; // multispectral data
float * lab; // initial labels

// parallelism stuff
pthread_attr_t attr; // specify threads joinable
pthread_mutex_t next_j_mtx; // lock for thread synchronization

size_t dst, src;
size_t min_idx_i, min_idx_j; // idx of segs to merge

vector<vector<size_t>> pix;
vector<priority_queue<f_i>> dmat; // distance matrix: self-sorting rows
size_t next_j, nr, nc, nb, np, max_label; // next job, row count, col count, band count, pixel count, max_label

inline void distance(size_t & j, size_t & i, size_t & np_j, size_t & np_i, size_t & k, size_t & m, size_t & n, size_t & pk, float & dij, float &d){
  d = 0.;

  // iterate points in segment j

  for0(m, np_j){
    size_t pj_m = pix[j][m]; // index of m-th point in seg j

    //iterate points in segment i
    for0(n, np_i){
      size_t pi_n = pix[i][n]; // index of n-th point in seg i

      // distance between points pi_n and pj_m
      dij = 0.;

      // iterate multispec bands
      for0(k, nb){
        pk = np * k; // band offset
        float dx = dat[pk + pi_n] - dat[pk + pj_m];
        dij += dx * dx;
      }
      d += sqrt(dij);
    }
  }
  // cprint(to_string(np_j) + str(",") + to_string(np_i) + str(",") + to_string(((float)np_j * (float)np_i)));

  size_t npji = np_j * np_i;
  if(npji > 0) d /= (float)npji;
}

// the activity for each thread
void * dmat_j(void * arg){
  float d, dx, dij;
  size_t i, j, k, m, n, pk;
  long w = (long)arg; // work index
  cprint(str("init work ") + std::to_string(w));

  while(1){
    // pick up a job
    mtx_lock(&next_j_mtx);
    j = next_j ++;
    d_count += 1;
    mtx_unlock(&next_j_mtx);
    size_t np_j = pix[j].size();

    // don't process if no pixels
    if(np_j == 0) continue;

    if(j > max_label){
      cprint(str("\texit work ") + to_string(w));
      return(NULL);
    }

    if(j % 100 == 0){
      float pct = 100. * (float)j / (float) max_label;
      cprint(str(" work ") + to_string(w) + str(" job ") + to_string(j) + str(" % ") + to_string(pct));
    }

    // pointer to dmat row for this element
    priority_queue<f_i> * pq = &dmat[j];

    // iterate over segments other than this one (j)
    for0(i, j){
      // max_label + 1)
      size_t np_i = pix[i].size();
      if(i != j && np_i > 0){

        // dist between segment j and i
        distance(j, i, np_j, np_i, k, m, n, pk, dij, d);

        if(d > 10000.){
          cprint(str("\td: ") + to_string(d) + str(" i,j=") + to_string(i) + str(",") + to_string(j));
        }

        // add distance and index of segment compared to
        pq->push(f_i(d, i));
      }
    }
  }
}
void deploy_works(void *(*work)(void *)){
  next_j = d_count = 0;
  int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
  // cout << "Number of cores: " << numCPU << endl;

  // mutex setup
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&next_j_mtx, NULL);

  // make the threads joinable
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // allocate threads
  pthread_t * my_pthread = new pthread_t[numCPU];
  unsigned int j;
  for0(j, numCPU){
    long k = j;
    pthread_create(&my_pthread[j], &attr, *work, (void *) k);
  }

  // wait for threads to finish
  for0(j, numCPU) pthread_join(my_pthread[j], NULL);
}

// the activity for each thread
void * dmat_j_update(void * arg){
  /* clean dmat row (j):
  for each row(j):
  A) if it's src, clear it
  B) if it's dst, clear it and recalc dist with respect to "everything" (under j); other elmt's reside in other rows
  C) if it's neither src nor dst, delete: all references to src or dst: and, recalculate any deleted elements */
  bool debug = false;
  float d, dx, dij;
  size_t i, j, k, m, n, pk, np_j, np_i;
  long w = (long)arg; // work index
  if(debug) cprint(str("init work ") + std::to_string(w));

  while(1){
    // pick up a job
    mtx_lock(&next_j_mtx);
    j = next_j ++;
    d_count += 1;
    mtx_unlock(&next_j_mtx);
    np_j = pix[j].size();

    // A) clear dmat[j] if seg-j empty, or if should be empty (j== src)
    if(np_j == 0 && j == src){
      while(dmat[j].size() > 0) dmat[j].pop();
      continue;
    }

    if(j > max_label){
      if(debug) cprint(str("\texit work ") + to_string(w));
      return(NULL);
    }

    if(debug && (j % 100 == 0)){
      float pct = 100. * (float)j / (float) max_label;
      cprint(str(" work ") + to_string(w) + str(" job ") + to_string(j) + str(" % ") + to_string(pct));
    }

    // pointer to dmat row for this element
    priority_queue<f_i> * pq = &dmat[j];

    // B) if it's dst, clear dmat[j] and recalc with respect to "everything" below j
    if(j == dst){
      while(dmat[j].size() > 0) dmat[j].pop(); // clear row

      // iterate over segments below j
      for0(i, j){
        // max_label + 1)
        np_i = pix[i].size();
        if(i != j && np_i > 0){

          // dist between segment j and i
          distance(j, i, np_j, np_i, k, m, n, pk, dij, d);

          if(debug && d > 10000.){
            cprint(str("\td: ") + to_string(d) + str(" i,j=") + to_string(i) + str(",") + to_string(j));
          }

          // add distance and index of segment compared to
          pq->push(f_i(d, i));
        }
      }
    }
    else{
      // move row to tmp
      priority_queue<f_i> tmp;
      if(tmp.size() > 0) err("sanity check failed");

      while(pq->size() > 0){
        tmp.push(pq->top());
        pq->pop();
      }
      if(pq->size() != 0) err("err"); // sanity check

      // process the items
      while(tmp.size() > 0){
        if(tmp.top().i == src){
          // drop
          if(debug) cout << j << "\tdrop " << tmp.top().i << " d=" << tmp.top().d << endl;
          // this distance represented w.r.t. dst now..
        }
        else if(tmp.top().i == dst){
          // recalc
          i = dst;
          np_i = pix[i].size();
          if(debug) cout << j << "\trclc " << tmp.top().i << " d=" << tmp.top().d;

          // apart from j, i, np_j, np_i, the other vars are set internally
          distance(j, i, np_j, np_i, k, m, n, pk, dij, d);

          if(debug) cout << " d'=" << d << endl;

          pq->push(f_i(d, i));
        }
        else{
          // keep
          pq->push(tmp.top());
        }
        tmp.pop();
      }
    }

  }
}

int main(int argc, char** argv){
  if(argc < 3) err(str("hclust [input file data] [label file] [desired number of clusters]") +
  str("hclust [input file data] [desired number of clusters]"));

  size_t i, j, k, min_label;
  size_t nr2, nc2, nb2, nclust;
  dat = load_envi(argv[1], nr, nc, nb);
  np = nr * nc; // pixel count

  if(argc < 4){
    //reduced args
    nr2 = nr;
    nc2 = nc;
    nb2 = nb;
    nclust = (size_t) atoi(argv[2]);
    lab = falloc(np);
    for0(i, np) lab[i] = i + 1;
  }
  else{
    //full args
    lab = load_envi(argv[2], nr2, nc2, nb2);
    if(nr != nr2 || nc != nc2) err("image size mismatch");
    if(nb2 != 1) err("second file should be 1-band label file");
    nclust = (size_t) atoi(argv[3]);
  }

  cout << "building set list.." << endl;
  for0(i, np){
    float f = lab[i];
    j = (size_t)f;
    if((float)j != f) err("int-float conversion failed");
    if(i == 0){
      min_label = max_label = f;
    }
    else{
      if(j < min_label) min_label = j;
      if(j > max_label) max_label = j;
    }
  }
  cout << "min_label: " << min_label << " max_label: " << max_label << endl;

  // pix[i]: list of indices of pixels, corresponding to label i
  vector<vector<size_t>>::iterator pi;
  for0(i, max_label + 1){
    pix.push_back(vector<size_t>());
    dmat.push_back(priority_queue<f_i>());
  }

  for0(i, np){
    j = (size_t)lab[i];
    pix[j].push_back(i);
  }

  // list of labels presently under consideration
  i = 0;
  list<size_t> labels;
  for(pi = pix.begin(); pi != pix.end(); pi++){
    if(pi->size() > 0) labels.push_back(i);
    i++;
  }

  labels.sort();
  cout << "labels:" << labels << endl;

  deploy_works(dmat_j);
  size_t expected = ((max_label * max_label - max_label) / 2);
  cout << "max_label " << max_label << " max_label^2 " << max_label * max_label << " max_label ^2 - max_label / 2 " << expected << endl;
  d_count = 0;
  vector<priority_queue<f_i>>::iterator dmi;
  float d_min = FLT_MAX;
  float d_max = FLT_MIN;
  float d_avg = 0.;
  size_t ci = 0;
  for(dmi = dmat.begin(); dmi != dmat.end(); dmi++){
    d_count += dmi->size();
    if(dmi->size() > 0){
      float d = dmi->top().d;
      if(d < d_min) d_min = d;
      if(d > d_max) d_max = d;
    }
  }
  cout << "dcount " << d_count << "d_min " << d_min << " d_max " << d_max << endl;
  if(d_count != expected) err("unexpected distance count");

  size_t iter = 0;
  cout << "labels.size() n_classes " << labels.size() << endl;
  while(labels.size() > nclust){

    // find the two closest segments, collecting min dist from each pq
    i = 0;
    min_idx_i = min_idx_j = 0;
    float min_dist = 0;
    list<size_t>::iterator it;
    for(it = labels.begin(); it != labels.end(); it++){
      j = *it;
      priority_queue<f_i> * pq = &dmat[j];
      if(pq->size() == 0) continue;
      if(i == 0){
        min_idx_j = j;
        min_dist = pq->top().d;
      }
      else{
        if(pq->top().d < min_dist){
          min_dist = pq->top().d;
          min_idx_j = j;
        }

      }
      i ++;
    }

    // indices of segs to merge
    min_idx_i = dmat[min_idx_j].top().i;
    cout << "iter= " << iter++ << " merge i,j= " << min_idx_i << "," << min_idx_j << " d=" << min_dist << " n_seg " << labels.size() << endl;
    // merge the points of the two clusters, into (by convention) the cluster with the smaller seg-index
    dst = (min_idx_i < min_idx_j)? min_idx_i: min_idx_j;
    src = (min_idx_i < min_idx_j)? min_idx_j: min_idx_i;
    // cout << "src" << pix[src] << endl << "dst" << pix[dst] << endl;

    // add src list elements onto dst list
    vector<size_t>::iterator pi;
    for(pi = pix[src].begin(); pi != pix[src].end(); pi++){
      pix[dst].push_back(*pi);
    }
    pix[src].clear(); // src list now defunct, clear it
    labels.remove(src); // remove defunct label from list (do we even use this?) probably not
    // cout << "src" << pix[src] << endl << "dst" << pix[dst] << endl;

    // update distane matrix
    deploy_works(dmat_j_update);

    // output labels
    for0(i, np){
      lab[i] = 0.;
    }
    i = 0;
    for(vector<vector<size_t>>::iterator pj = pix.begin(); pj != pix.end(); pj++){
      for(vector<size_t>::iterator pi = (*pj).begin(); pi != (*pj).end(); pi++){
        lab[*pi] = i;
      }
      i ++;
    }

    str ofn(str(argv[1]) + str("_iter_") + to_string(iter) + ".bin");
    str ohf(str(argv[1]) + str("_iter_") + to_string(iter) + ".hdr");
    hwrite(ohf, nr, nc, 1);
    FILE * of = wopen(ofn);
    fwrite(lab, sizeof(float), np, of);
    fclose(of);

    if(iter % 50 == 0){

    str cmd(str("bin/class_recode ") + ofn);
    cout << cmd << endl;
    system(cmd.c_str());

    str cm2(str("bin/class_wheel ") + ofn + str("_recode.bin"));
    cout << cm2 << endl;
    system(cm2.c_str());

      str cm3(str("python py/read_multi.py ") + ofn + str("_recode.bin_wheel.bin") + str(" 1"));
      cout << cm3 << endl;
      system(cm3.c_str());
    } 
  }
  return 0;
}
