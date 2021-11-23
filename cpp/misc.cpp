#include"misc.h"

string cwd(){
  char s[PATH_MAX];
  char * ss = _cwd(s, PATH_MAX);
  return string(s);
}

/* split a string (a-la python) */
vector<string> split(string s, char delim){
  trim(s);
  bool extra = (s[s.size() - 1] == delim);
  std::vector<std::string> ret;
  std::istringstream iss(s);
  std::string token;
  while(getline(iss,token,delim)) ret.push_back(token);

  if(extra) ret.push_back(str(""));
  return ret;
}

/* split a string (a-la python) */
vector<string> split(string s){
  trim(s);
  const char delim = ',';
  bool extra = (s[s.size() -1] == delim);

  std::vector<std::string> ret;
  std::istringstream iss(s);
  string token;
  while(getline(iss,token,delim)) ret.push_back(token);

  vector<string>::iterator it;
  for(it = ret.begin(); it != ret.end(); it++){
    trim(*it);
  }
  if(extra) ret.push_back(str(""));
  return ret;
}

void dbg(char c, int start_pos, int end_pos, str action, bool inside_quotes){
  cout << "c[" << c << "] start [" << start_pos << "] end [" << end_pos << "] " << action << " " << (inside_quotes?str("inside"):str("outside")) << "]" <<endl;
}

/*
#include"misc.cpp"
int main(int argc, char ** argv){
  str data("1,\",,2,3\",,,hello,12345,\"{12,12,12}\",word");
  cout << "[" << data << "]" << endl;
  cout << split(data) << endl;
  vector<str> ss(split_special(data));
  cout << "good:"<< ss << endl;
  cout << "bad:" << split(data) << endl;
  cout << data << endl;
  return 0;
}
*/

vector<string> split_special(string s){
  // split string function for csv files that may have commas inside of double quotation marks!
  vector<string> ret;
  if(s.size() == 0) return ret;
  int start_pos = 0;
  int end_pos = 0; // left-inclusive indices of selection (right index is not inclusive)
  bool inside_quotes = false;
  char c = s[end_pos];
  while(end_pos < s.size() - 1){
    while(c != ',' && c != '\"' && end_pos < s.size() -1){
      c = s[++end_pos];
    }
    // hit a comma, a double-quotes mark, or got to the end
    if(c == ','){
      if(!inside_quotes){
        if(end_pos > start_pos){
          str add(s.substr(start_pos, end_pos - start_pos));
          trim(add, '\"');
          std::replace(add.begin(), add.end(), ',', ';');
          ret.push_back(add);
        }
        else{
          ret.push_back(str(""));
        }
        start_pos = ++end_pos;
        c = s[end_pos];
      }
      else{
        c = s[++end_pos];
      }
    }
    else if(c == '\"'){
      inside_quotes = inside_quotes?false:true;
      c = s[++end_pos];
    }
    else{
      // out of bounds
      str add(s.substr(start_pos, end_pos - start_pos + 1));
      trim(add, '\"');
      std::replace(add.begin(), add.end(), ',', ';');
      ret.push_back(add);
      return ret;
    }
  }
  return ret;
}

/*
e.g.:
str join_(vector<str> d){
  const char * d = "_\0";
  return join(d, ans);
}
*/
string join(const char * delim, vector<string> s){
  string ret("");
  string d(delim);
  for(vector<string>::iterator it = s.begin(); it!=s.end(); it++){
    if(it!=s.begin()) ret += d;
    ret += *it;
  }
  return ret;
}

void err(string msg){
  cout << "Error: " << msg << endl;
  exit(1);
}

void err(const char * msg){
  err(string(msg));
}

/*
void * balloc(long unsigned int nb){
  void * d = malloc(nb);
  memset(d, '\0', nb);
  return (void *)d;
}
*/

FILE * wopen(string fn){
  FILE * f = fopen(fn.c_str(), "wb");
  if(!f) {
    printf("Error: failed to open file for writing: %s\n", fn.c_str());
    err("failed to open file for writing");
  }
  return f;
}

FILE * wopen(const char * fn){
  return wopen(str(fn));
}

FILE * ropen(string fn){
  FILE * f = fopen(fn.c_str(), "rb");
  if(!f){
    printf("Error: failed to open file for reading: %s\n", fn.c_str());
    err("failed to open file for reading");
  }
  return f;
}

FILE * ropen(const char * fn){
  return ropen(str(fn));
}

/* get size of file pointer */
size_t size(FILE * f){
  fseek(f, 0L, SEEK_END);
  size_t sz = ftell(f);
  rewind(f);
  return sz;
}

/* get file size */
size_t fsize(string fn){
  FILE * f = fopen(fn.c_str(), "rb");
  if(!f) return (size_t) 0;
  size_t fs = size(f);
  fclose(f);
  return fs;
}

/* get file size */
size_t fsize(const char * fn){
  FILE * f = fopen(fn, "rb");
  if(!f) return (size_t) 0;
  size_t fs = size(f);
  fclose(f);
  return fs;
}

void rewind(ifstream &a){
  a.clear();
  a.seekg(0);
}

bool operator<(const f_idx& a, const f_idx&b){
  return a.d > b.d; // priority_queue max first: we want min first
}

// read header file
size_t hread(str hfn, size_t & nrow, size_t & ncol, size_t & nband){
  str line;
  size_t data_type;
  vector<str> words;
  ifstream hf(hfn);
  data_type = nrow = ncol = nband = 0;
  if(hf.is_open()){
  }
  else{
    err(str("failed to open header file: ") + hfn);
  }
  while(getline(hf, line)){
    words = split(line, '=');
    if(words.size() == 2){
      strip(words[0]);
      str w(words[0]);
      size_t n = (size_t) atoi(words[1].c_str());
      if(w == str("samples")) ncol = n;
      if(w == str("lines")) nrow = n;
      if(w == str("bands")) nband = n;
      if(w == str("data type")) data_type = n;
    }
  }
  cout << "hread: " << hfn << " nrow: " << nrow << " ncol: " << ncol << " nband: " << nband << " data_type: " << data_type << endl;
  hf.close();
  return data_type;
}

void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband){
  cout << "hwrite" << endl;
  cout << "+w " << hfn << " nrow " << nrow << " ncol " << ncol << " nband " << nband << endl;
  ofstream hf(hfn);
  if(!hf.is_open()) err("failed to open header file for writing");
  hf << "ENVI" << endl;
  hf << "samples = " << ncol << endl;
  hf << "lines = " << nrow << endl;
  hf << "bands = " << nband << endl;
  hf << "header offset = 0" << endl;
  hf << "file type = ENVI Standard" << endl;
  hf << "data type = 4" << endl;
  hf << "interleave = bsq" << endl;
  hf << "byte order = 0" << endl;
  hf << "band names = {band 1";
  size_t i;
  for0(i, nband - 1){
    hf << ",\n" << "band " << i + 2;
  }
  hf << "}" << endl;
  hf.close();
}

void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband, size_t data_type){
  cout << "hwrite" << endl;
  cout << "+w " << hfn << " nrow " << nrow << " ncol " << ncol << " nband " << nband << endl;
  ofstream hf(hfn);
  if(!hf.is_open()) err("failed to open header file for writing");
  hf << "ENVI" << endl;
  hf << "samples = " << ncol << endl;
  hf << "lines = " << nrow << endl;
  hf << "bands = " << nband << endl;
  hf << "header offset = 0" << endl;
  hf << "file type = ENVI Standard" << endl;
  hf << "data type = " << data_type << endl;
  hf << "interleave = bsq" << endl;
  hf << "byte order = 0" << endl;
  hf << "band names = {band 1";
    size_t i;
  for0(i, nband - 1){
    hf << ",\n" << "band " << i + 2;
  }
  hf << "}" << endl;
  hf.close();
}

str hdr_fn(str fn, bool create=false){
  str hfn(fn + str(".hdr"));
  if(exists(hfn)) return hfn;
  int s = fn.size();
  str b(fn);
  b = b.substr(0, s - 4);
  //cout << "b " << b << endl;
  str hfn2(b + str(".hdr"));
  //cout << "hfn2 " << hfn2 << endl;
  if(exists(hfn2)){
    return hfn2; // don't create two headers for the same file
  }
  if(create== true){
    return hfn2;
  }
  else{
    err(str("could not find header at: ") + hfn + str(" or at: ") + hfn2);
  }
  return(hfn);
}

str hdr_fn(str fn){
  return hdr_fn(fn, false);
}

float * falloc(size_t nf){
  return (float *) alloc(nf * (size_t)sizeof(float));
}

double * dalloc(size_t nd){
  return (double *) alloc(nd * (size_t)sizeof(double));
}

// read binary file
float * bread(str bfn, size_t nrow, size_t ncol, size_t nband){
  FILE * f = fopen(bfn.c_str(), "rb");
  size_t nf = nrow * ncol * nband;
  float * dat = falloc(nf);
  size_t nr = fread(dat, nf * (size_t)sizeof(float), 1, f);
  if(nr != 1){
    printf("bread(%s)\n", bfn.c_str());
    err("misc.cpp: bread(): failed to read data");
  }
  fclose(f);
  return dat;
}

void bwrite(float * d, str bfn, size_t nrow, size_t ncol, size_t nband){
  size_t nf = nrow * ncol * nband;
  FILE * f = fopen(bfn.c_str(), "wb");
  if(!f) err("bwrite(): failed to open output file");
  fwrite(d, sizeof(float), nf, f);
  fclose(f);
}

pthread_mutex_t print_mtx;

void cprint(str s){
  mtx_lock(&print_mtx);
  cout << s << endl;
  mtx_unlock(&print_mtx);
}

int hsv_to_rgb(float *r, float *g, float *b, float h, float s, float v){
  if( (h>360.)||(h<0.)){
    printf("H: HSV out of range %f %f %f\n", h, s, v);
    return(1);
  }
  if((s<0.)||(s>1.)){
    printf("S: HSV out of range %f %f %f\n", h, s, v);
    return(1);
  }
  if((v<0.)||(v>1.)){
    printf("V: HSV out of range %f %f %f\n", h, s, v);
    return(1);
  }
  if (h==360.){
    h=0;
  }
  int i;
  float f, p, q, t;
  if( s == 0 ) {
    // achromatic (grey)
    *r = *g = *b = v;
    return 0;
  }
  float H,S,V;
  H=h; V=v; S=s;
  h /= 60.; // sector 0 to 5
  i = (int)floor( h );
  f = h - i; // factorial part of h
  p = v * ( 1. - s );
  q = v * ( 1. - s * f );
  t = v * ( 1. - s * ( 1 - f ) );
  switch( i ) {
    case 0: *r = v; *g = t; *b = p; break;
    case 1: *r = q; *g = v; *b = p; break;
    case 2: *r = p; *g = v; *b = t; break;
    case 3: *r = p; *g = q; *b = v; break;
    case 4: *r = t; *g = p; *b = v; break;
    case 5: *r = v; *g = p; *b = q; break;
    default: printf("\nERROR HSV to RGB"); printf("i=%d hsv= %f %f %f\n", i, H, S, V);
    //exit(1);
  }
  return 0;
}

/*
size_t size(FILE * f){
  fseek(f, 0L, SEEK_END);
  size_t sz = ftell(f);
  rewind(f);
  return sz;
}

size_t fsize(string fn){
  FILE * f = fopen(fn.c_str(), "rb");
  if(!f) return (size_t) 0;
  size_t fs = size(f);
  fclose(f);
  return fs;
}
*/

bool exists(str fn){
  return fsize(fn) > 0;
}

bool operator<(const f_i& a, const f_i&b){
  return a.d > b.d; // priority_queue max first: we want min first
}

bool operator<(const f_ij& a, const f_ij&b){
  return a.d > b.d; // priority_queue max first: we want min first
}

float * load_envi(str in_f, size_t & nrow, size_t & ncol, size_t & nband){
  str ext(in_f.substr(in_f.size() - 3, 3));
  if(ext != str("bin")){
    err(".bin extension expected on input file");
  }
  str hdr(in_f.substr(0, in_f.size() - 3) + str("hdr"));
  if(!exists(hdr)){
    hdr = in_f + str(".hdr");
    if(!exists(hdr)){
      err(str("header file not found:") + hdr);
    }
  }

  hread(hdr, nrow, ncol, nband);
  cout << hdr << " nrow " << nrow << " ncol " << ncol << " nband " << nband << endl;

  return bread(in_f, nrow, ncol, nband);
}

str zero_pad(str x, int n_zero){
  return std::string(n_zero - x.length(), '0') + x;
}

void int_write(size_t value, str fn){
  FILE * f = wopen(fn);
  fwrite(&value, sizeof(size_t), 1, f);
  fclose(f);
}

size_t int_read(str fn){
  size_t value;
  FILE * f = ropen(fn);
  size_t nr = fread(&value, sizeof(size_t), 1, f);
  fclose(f);
  return value;
}

size_t * ints_read(str fn){
  size_t fs = fsize(fn);
  size_t * value = (size_t *)alloc(fs);
  FILE * f = ropen(fn);
  size_t br = fread(value, 1, fs, f);
  if(br != fs) err("unexpected byte read count");
  return value;
}


void float_write(float * d, size_t n, str fn){
  FILE * f = wopen(fn);
  size_t nr = fwrite(d, sizeof(float), n, f);
  if(nr != n){
    printf("nr=%zu != n= %zu\n", nr, n);
    err("unexpected write length");
  }
  fclose(f);
}

float * float_read(str fn){
  size_t fs = fsize(fn);
  size_t n = fs / sizeof(float);
  float * dat = falloc(n);
  FILE * f = ropen(fn);
  size_t nr = fread(dat, sizeof(float), n, f);
  fclose(f);
  return(dat);
}

float * float_read(str fn, size_t &n){
  size_t fs = fsize(fn);
  n = fs / sizeof(float);
  float * dat = falloc(n);
  FILE * f = ropen(fn);
  size_t nr = fread(dat, sizeof(float), n, f);
  fclose(f);
  return(dat);
}

// parallelism stuff: which variables do we actually need?
pthread_attr_t pt_attr; // specify threads joinable
pthread_mutex_t pt_nxt_j_mtx; // work queue
size_t pt_nxt_j; // next job to run
size_t pt_start_j;
size_t pt_end_j; // start and end indices for job
void (*pt_eval)(size_t); // function pointer to execute in parallel, over range start_j:end_j inclusive

void pt_init_mtx(){
  // mutex setup
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pt_nxt_j_mtx, NULL);
}

void * pt_worker_fun(void * arg){
  size_t k, my_nxt_j; // pthread "worker" function for parfor
  k = (size_t)arg; // cprint(str("worker_fun(") + std::to_string(k) + str(")"));
  while(1){
    mtx_lock(&pt_nxt_j_mtx); // try to pick up a job
    my_nxt_j = pt_nxt_j ++; // index of data this thread should pick up if it can
    mtx_unlock(&pt_nxt_j_mtx);
    if(my_nxt_j >= pt_end_j) return(NULL);  // cprint(str("\texit thread ") + to_string(k));
    //if(my_nxt_j % 10000 == 0) cprint(to_string(my_nxt_j));
    pt_eval(my_nxt_j); // perform action segment
  }
}

void parfor(size_t start_j, size_t end_j, void(*eval)(size_t)){
  pt_eval = eval; // set global function pointer
  pt_end_j = end_j;
  pt_nxt_j = start_j; // pt_nxt_j_mtx locks this variable
  size_t j, n_cores;
  pt_start_j = start_j;
  n_cores = sysconf(_SC_NPROCESSORS_ONLN); // cout << "Number of cores: " << n_cores << endl;
  pthread_attr_init(&pt_attr); // allocate threads, make threads joinable
  pthread_attr_setdetachstate(&pt_attr, PTHREAD_CREATE_JOINABLE);
  pthread_t * my_pthread = new pthread_t[n_cores];
  for0(j, n_cores) pthread_create(&my_pthread[j], &pt_attr, pt_worker_fun, (void *)j);
  for0(j, n_cores) pthread_join(my_pthread[j], NULL); // wait for threads to finish
  // cprint(str("return parfor()"));
  delete my_pthread;
}

// parameters always named (in json-like format)?
