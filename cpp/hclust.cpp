/*
dent cpp/hclust.cpp; vim cpp/hclust.cpp; rm bin/hclust; python setup.py; ./bin/hclust data/fran/mS2.bin data/fran/mS2.bin_label.bin 10
dent cpp/hclust.cpp; vim cpp/hclust.cpp; rm bin/hclust; python setup.py; ./bin/hclust data/fran/mS2.bin 10


20191201:

python py/fast_cluster_tiling.py  data/fran/mS2.bin 70 10 24.8
bin/hclust data/fran/mS2.bin data/fran/mS2.bin_label.bin 15

*/

#include"misc.h"

float * dat; // multispectral data
float * lab; // initial labels

pthread_attr_t attr; // specify threads joinable
pthread_mutex_t nxt_j_mtx; // lock for thread synchronization

size_t d_count; // count of distance calcs
size_t dst, src; // index of seg to merge into, and from
size_t min_idx_i, min_idx_j; // idx of segs to merge

vector<vector<size_t>> pix; // list of pixels, as a function of label
vector<priority_queue<f_i>> dmat; // distance matrix: self-sorting rows
size_t next_j, nr, nc, nb, np, max_label; // next job, row count, col count, band count, pixel count, max_label

inline void distance(size_t & j, size_t & i, size_t & np_j, size_t & np_i, size_t & k, size_t & m, size_t & n, size_t & pk, float & dij, float &d){
  d = 0.;

  for0(m, np_j){
    size_t pj_m = pix[j][m]; // index of m-th point in seg j

    for0(n, np_i){
      size_t pi_n = pix[i][n]; // index of n-th point in seg i

      dij = 0.; // distance between pixels at index pi_n, pj_m

      for0(k, nb){
        pk = np * k; // multispec band offset
        float dx = dat[pk + pi_n] - dat[pk + pj_m];
        dij += dx * dx;
      }
      d += sqrt(dij); // add the distance for each pair of points
    }
  }

  // convert total to average
  size_t npji = np_j * np_i;
  if(npji > 0.) d /= (float)npji;
}

void * dmat_j(void * arg){
  // thread activity: distance matrix row

  float d, dx, dij;
  size_t i, j, k, m, n, pk;
  long w = (long)arg; // work index
  cprint(str("init work ") + std::to_string(w));

  // keep picking up jobs
  while(1){

    mtx_lock(&nxt_j_mtx);
    j = next_j ++;
    d_count ++;
    mtx_unlock(&nxt_j_mtx);

    if(j > max_label){
      cprint(str("\texit work ") + to_string(w));
      return(NULL);
    }

    size_t np_j = pix[j].size();

    // don't process if no pixels
    if(np_j == 0) continue;

    if(j % 100 == 0){
      float pct = 100. * (float)j / (float)max_label;
      cprint(str(" work ") + to_string(w) + str(" job ") + to_string(j) + str(" % ") + to_string(pct));
    }

    // pointer to dmat row for this element
    priority_queue<f_i> * pq = &dmat[j];

    // iterate over segments other than this one (j)
    for0(i, j){

      size_t np_i = pix[i].size();
      if(i != j && np_i > 0){

        distance(j, i, np_j, np_i, k, m, n, pk, dij, d); // distance between seg i and seg j

        pq->push(f_i(d, i)); // add distance and index of compared seg
      }
    }
  }
}
void deploy_works(void *(*work)(void *)){
  next_j = d_count = 0;

  int numCPU = sysconf(_SC_NPROCESSORS_ONLN); // cout << "Number of cores: " << numCPU << endl;

  // mutex setup
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&nxt_j_mtx, NULL);

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

void * dmat_j_update(void * arg){

  /* thread activity: update dmat row (j):

  for each row(j):
  A) if it's src, clear it
  B) if it's dst, clear it and recalc dist with respect to "everything" (under j); other elmt's reside in other rows
  C) if it's neither src nor dst, delete: all references to src or dst: and, recalculate any deleted elements */

  bool debug = false;
  float d, dx, dij;
  size_t i, j, k, m, n, pk, np_j, np_i;
  long w = (long)arg; // worker index
  if(debug) cprint(str("init work ") + std::to_string(w));

  // keep picking up jobs
  while(1){

    mtx_lock(&nxt_j_mtx);
    j = next_j ++;
    d_count += 1;
    mtx_unlock(&nxt_j_mtx);

    np_j = pix[j].size();

    if(np_j == 0 && j == src){
      // A) clear dmat[j] if seg-j empty, or if should be empty (j== src)
      while(dmat[j].size() > 0) dmat[j].pop();
      continue;
    }

    if(j > max_label){
      if(debug) cprint(str("\texit work ") + to_string(w));
      return(NULL);
    }

    if(debug && (j % 100 == 0)){
      float pct = 100. * (float)j / (float) max_label;
      cprint(str(" worker ") + to_string(w) + str(" job ") + to_string(j) + str(" % ") + to_string(pct));
    }

    // pointer to dmat row for this element
    priority_queue<f_i> * pq = &dmat[j];

    if(j == dst){
      // B) if it's dst, clear dmat[j] and recalc with respect to "everything" below j
      while(dmat[j].size() > 0) dmat[j].pop();

      for0(i, j){
        // iterate over segs below j
        np_i = pix[i].size();

        if(i != j && np_i > 0){
          // dist between segment j and i
          distance(j, i, np_j, np_i, k, m, n, pk, dij, d);

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

      while(tmp.size() > 0){
        // process items
        if(tmp.top().i == src){
          // drop item
          if(debug) cout << j << "\tdrop " << tmp.top().i << " d=" << tmp.top().d << endl;
          // that distance represented w.r.t. dst now
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
          // keep existing element
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
  dat = load_envi(argv[1], nr, nc, nb); // load raw-binary data

  np = nr * nc; // pixel count

  if(argc < 4){
    printf("reduced args: setting each pixel as a seed\n");
    nr2 = nr;
    nc2 = nc;
    nb2 = nb;
    nclust = (size_t) atoi(argv[2]);
    lab = falloc(np);
    for0(i, np) lab[i] = i + 1;
  }
  else{
    //full args
    printf("full args: seed labels supplied from: %s\n", argv[2]);
    lab = load_envi(argv[2], nr2, nc2, nb2);
    if(nr != nr2 || nc != nc2) err("image size mismatch");
    if(nb2 != 1) err("second file should be 1-band label file");
    nclust = (size_t) atoi(argv[3]);
  }

  // examine seeds
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

  // pix[i]: list of indices of pixels for label i
  vector<vector<size_t>>::iterator pi;
  for0(i, max_label + 1){
    pix.push_back(vector<size_t>());
    dmat.push_back(priority_queue<f_i>());
  }

  // build pixel list
  for0(i, np) pix[(size_t)lab[i]].push_back(i);

  // list of labels presently considered
  i = 0;
  list<size_t> labels;
  for(pi = pix.begin(); pi != pix.end(); pi++){
    if(pi->size() > 0){
      labels.push_back(i);
    }
    i++;
  }

  labels.sort();

  // calculate dmat
  deploy_works(dmat_j);

  // expected number of distance matrix elements
  size_t expected = ((max_label * max_label - max_label) / 2);
  cout << "max_label " << max_label << " max_label^2 " << max_label * max_label << " max_label ^2 - max_label / 2 " << expected << endl;

  // count number of distance matrix elements
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
  // if(d_count != expected) err("unexpected distance count");

  // track frames for movie
  str flist_fn("hclust_files.txt");
  ofstream flist(flist_fn);
  if(!flist.is_open()) err("failed to open framelist file");

  // start merging
  size_t iter = 1;
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

      // ignore labels without pixels
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
      i++;
    }

    // indices of segs to merge
    min_idx_i = dmat[min_idx_j].top().i;
    cout << "iter= " << iter++ << " merge i,j= " << min_idx_i << "," << min_idx_j << " d=" << min_dist << " n_seg " << labels.size() << endl;

    // merge the points of the two clusters, into (by convention) the cluster with the smaller seg-index
    dst = (min_idx_i < min_idx_j)? min_idx_i: min_idx_j;
    src = (min_idx_i < min_idx_j)? min_idx_j: min_idx_i;

    // add src list elements onto dst list
    vector<size_t>::iterator pi;
    for(pi = pix[src].begin(); pi != pix[src].end(); pi++) pix[dst].push_back(*pi);

    // clear now-defunct list and remove label
    pix[src].clear();
    labels.remove(src);

    // revise distance matrix
    deploy_works(dmat_j_update);

    // update labels
    for0(i, np) lab[i] = 0.;
    i = 0;
    vector<vector<size_t>>::iterator pj;
    for(pj = pix.begin(); pj != pix.end(); pj++){
      for(vector<size_t>::iterator pi = (*pj).begin(); pi != (*pj).end(); pi++){
        lab[*pi] = i;
      }
      i++;
    }

    if(iter % 50 == 0 || labels.size() == nclust){
    // output label maps. left zero-pad output filenames
    int n_zero = 6;
    str iter_s(to_string(iter));
    std::string iter_s2 = std::string(n_zero - iter_s.length(), '0') + iter_s;
    str ofn(str(argv[1]) + str("_iter_") + iter_s2 + ".bin");
    str ohf(str(argv[1]) + str("_iter_") + iter_s2 + ".hdr");
    hwrite(ohf, nr, nc, 1);
    FILE * of = wopen(ofn);
    fwrite(lab, sizeof(float), np, of);
    fclose(of);

    // color-code every n-th map

    // if(iter % 50 == 0 || labels.size() == nclust )
      str cmd(str("bin/class_recode ") + ofn);
      cout << cmd << endl;
      system(cmd.c_str());

      cmd = (str("bin/class_wheel ") + ofn + str("_recode.bin"));
      cout << cmd << endl;
      system(cmd.c_str());

      cmd = (str("python py/read_multi.py ") + ofn + str("_recode.bin_wheel.bin") + str(" 1"));
      cout << cmd << endl;
      system(cmd.c_str());

      flist << "file '" + ofn + str("_recode.bin_wheel.bin.png'") << endl;
    }
  }

  str mp4_fn(str(argv[1]) + "_iter.mp4");
  str cmd(str("ffmpeg -f concat -r 5 -i ") + flist_fn + str(" -shortest ") + mp4_fn);
  cout << cmd << endl;
  system(cmd.c_str());
  return 0;
}
