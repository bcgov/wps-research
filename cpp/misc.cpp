#include"misc.h"
/* run system command and collect results */
std::string exec(const char* cmd) {
  char buffer[128];
  std::string result = "";
  FILE* pipe = popen(cmd, "r");
  if (!pipe){
	  throw std::runtime_error("popen() failed!");
  }
  try{
    while(fgets(buffer, sizeof buffer, pipe) != NULL){
      result += buffer;
    }
  }
  catch (...){
    pclose(pipe);
    throw;
  }
  pclose(pipe);
  return result;
}

bool not_space(int data){
  return !std::isspace(data);
}

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
int main(int argc, char ** argv)
str data("1,\",,2,3\",,,hello,12345,\"{12,12,12}\",word");
cout << "[" << data << "]" << endl;
cout << split(data) << endl;
vector<str> ss(split_special(data));
cout << "good:"<< ss << endl;
cout << "bad:" << split(data) << endl;
cout << data << endl;
return 0;
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
  printf("+w %s\n", fn.c_str());
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
  //cout << "hwrite" << endl;
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

// need to clean these up:
void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband, size_t data_type){
  //cout << "hwrite" << endl;
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

void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband, size_t data_type, vector<string> & bandNames){
  //cout << "hwrite" << endl;
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
  hf << "band names = {";
  hf << bandNames[0];
  size_t i;
  for0(i, nband - 1){
    hf << ",\n" << bandNames[i + 1];
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

// read bsq binary file (assumed float)
float * bread(str bfn, size_t nrow, size_t ncol, size_t nband){
  FILE * f = ropen(bfn);
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

//read binary file in char form
char * read(str bfn, size_t & n_bytes){
  n_bytes = fsize(bfn);
  char * d = (char *)(void *)alloc(n_bytes);
  FILE * f = ropen(bfn);
  size_t nr = fread(d, 1, n_bytes, f);
  if(nr != n_bytes){
    printf("Error: bytes expected %zu, read: %zu\n", n_bytes, nr);
    err("unexpected number of bytes read\n");
  }
  fclose(f);
  return d;
}

void bwrite(float * d, str bfn, size_t nrow, size_t ncol, size_t nband){
  size_t nf = nrow * ncol * nband;
  FILE * f = wopen(bfn.c_str()); //fopen(bfn.c_str(), "wb");
  if(!f) err("bwrite(): failed to open output file");
  fwrite(d, sizeof(float), nf, f);
  fclose(f);
}

/* append to binary file. Set start to true, to clear the file:
e.g. on a first iteration*/
void bappend(float * d, FILE * f, size_t n_float){
  size_t nw = fwrite(d, sizeof(float), n_float, f);
  if(nw != n_float){
    printf("wrote: %zu expected: %zu\n", nw, n_float);
    err("bappend: incorrect number of records written");
  }
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
    if(my_nxt_j >= pt_end_j) return(NULL); // cprint(str("\texit thread ") + to_string(k));
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
  delete [] my_pthread;
}

// parameters always named (in json-like format)?

vector< vector<str> > read_csv(str fn, vector<str> & hdr){
  vector< vector<str> > output; // read lines from csv file
  ifstream ifs(fn); // stream to input file
  str token;
  size_t ci = 0;
  while(getline(ifs, token, '\n')){
    vector<str> words(split(token, ','));
    // cout << words << endl;
    if(ci == 0) hdr = words;
    else output.push_back(words);
    ++ci;
  }
  return output; // n.b., we assumed CSV was simple and well-formed (no quotes, same number of fields per line, etc).
}

vector<string> parseHeaderFile(string hfn, size_t & NRow, size_t & NCol, size_t & NBand){
  vector<string> bandNames;
  if(!exists(hfn)){
    printf("%sError: couldn't find header file\n", KRED);
  }
  else{
    vector<string> lines(readLines(hfn));
    vector<string>::iterator it;
    for(it=lines.begin(); it!=lines.end(); it++){
      string sss(*it);
      vector<string> splitLine(split(sss, '='));
      if(splitLine.size()==2){
        if(strncmp(strip_space(splitLine[0]).c_str(), "samples", 7) == 0){
          NCol = atoi(strip_space(splitLine[1]).c_str());
        }
        if(strncmp(strip_space(splitLine[0]).c_str(), "lines", 5) == 0){
          NRow = atoi(strip_space(splitLine[1]).c_str());
        }
        if(strncmp(strip_space(splitLine[0]).c_str(), "bands", 5)== 0){
          NBand = atoi(strip_space(splitLine[1]).c_str());
        }
        if(strncmp(strip_space(splitLine[0]).c_str(), "band names", 10) == 0){
          string bandNameList(trim2(trim2(strip_space(splitLine[1]),
          ('{')),
          ('}')));
          bandNames = split(bandNameList, ',');
        }
      }
    }
  }
  return bandNames;
}

vector<string> readLines(string fn){
  vector<string> ret;
  size_t fs = fsize(fn);
  char * fd = (char *)(void *)malloc(fs);
  memset(fd, '\0',fs);
  FILE * f = fopen(fn.c_str(), "rb");
  size_t br = fread(fd, fs, 1, f);
  fclose(f);
  // free(fd);
  ret = split(fd, fs, '\n');
  return(ret);
}

/* split a string (a-la python) */
/*vector<string> split(string s, char delim){
  vector<string> ret;
  size_t N = s.size();
  if(N == 0) return ret;
  if(delim == '\n'){
    return(split(s, delim, 0));
  }
  istringstream iss(s); string token;
  while(getline(iss,token,delim)){
    ret.push_back(token);
  }
  return ret;
}
// should probably compare this with the above..
*/

// need to replace string with SA concept, to make this work! Use "binary string" instead of string
vector<string> split(char * s, size_t s_len, char delim){
  // needed to write a special split, for string that might contain newlines / might be a file. In the future we should subsume string, array and file with a generic class like we did in meta4

  // printf("split()\n");
  vector<string> ret;
  string ss("");
  size_t i = 0;
  while(i < s_len){
    if(s[i] == delim){
      ret.push_back(ss);
      //cout << "\t" << ss << endl;
      ss = "";
    }
    else{
      ss += s[i];
    }
    i++;
  }
  if(ss.length() > 0){
    ret.push_back(ss);
    //cout << "\t" << ss << endl;
    ss = "";
  }
  return ret;
}

/*
vector<string> split(string s){
  return split(s, ' ');
}
*/

/*trim leading or trailing characters from a string*/
string trim2(string s, char a){
  string ret("");
  size_t i, j, N;
  N = s.size();
  if(N == 0){
    return s;
  }
  i = 0;
  while(i < N && (s[i] == a)){
    i++;
  }
  j = N - 1;
  while(j > i && (s[j] == a)){
    j--;
  }
  for(N = i; N <= j; N++){
    ret = ret + chartos(s[N]);
  }
  return ret;
}

/*convert char to string: single character: interpret whitspace as space character */
string chartos(char s){
  string ret("");
  stringstream ss;
  ss << s;
  ss >> ret;
  if(isspace(s)){
    ret += " ";
  }
  return ret;
}

/*strip leading or trailing whitespace from a string*/
string strip_space(string s){
  string ret("");
  size_t i, j, N;
  N = s.size();
  if(N == 0) return s;
  i = 0;
  while(i < N && isspace(s[i])){
    i++;
  }
  j = N-1;
  while(j > i && isspace(s[j])){
    j--;
  }
  for(N = i; N <= j; N++){
    ret = ret + chartos(s[N]);
  }
  return ret;
}

vector<string> parse_band_names(string fn){
  size_t nr, nc, nb;
  if(!exists(fn)) err("parse_band_names: header file not found");
  hread(fn, nr, nc, nb);

  str band("band");
  str names("names");
  size_t ci = 0;
  size_t bni = -1; // band names start index
  vector<string> lines(readLines(fn)); // read header file
  for(vector<string>::iterator it = lines.begin(); it != lines.end(); it++){
    //cout << "\t" << *it << endl;
    vector<string> w(split(*it, ' '));
    /*
    for(vector<string>::iterator it2 = w.begin(); it2 != w.end(); it2++){
      cout << "\t\t" << *it2 << endl;
    }
    */
    // cout << w << endl;
    if(w.size() >= 2){
      if((w[0] == band) && (w[1] == names)){
        bni = ci;
        break;
      }
    }
    ci ++;
  }

  vector<string> band_names;
  vector<string> w(split(lines[bni], '{')); // parse first band name
  string bn(w[1]);
  bn = trim2(bn, ',');
  //cout << bn << endl;
  band_names.push_back(bn);

  if(nb > 1){

    if(nb > 2){
      // parse middle band names
      for(ci = bni + 1; ci < lines.size() - 1; ci++){
        str line(lines[ci]);
        line = trim2(line, ',');
        //cout << line << endl;
        band_names.push_back(line);
      }
    }

    // parse last band name
    if(true){
      str w(lines[lines.size() -1]);
      w = trim2(w, '}');
      band_names.push_back(w);
    }
  }
  for0(ci, band_names.size()){
    trim(band_names[ci], '}');
  }
  return band_names;
}

size_t hread(str hfn, size_t & nrow, size_t & ncol, size_t & nband, vector<string>& bandNames){
  size_t ret = hread(hfn, nrow, ncol, nband);
  bandNames = parse_band_names(hfn);
  return ret;
}

bool contains(string s1, string s2){
  // does one string contain another as substring?
  // cout << "\tcontains(" << s1 << "," << s2 << ")=" << (s1.find(s2) != std::string::npos) << endl;
  return s1.find(s2) != std::string::npos;
}

void status(size_t i, size_t of){
  cprint(to_string(100.* ((float)(i+1) / (float)of)) + str(" % ") +
  to_string(i) + str(" / ") + to_string(of));
}

void write_config(str fn, size_t nrow, size_t ncol){
  FILE * f = wopen(fn);
  fprintf(f, "Nrow\n");
  fprintf(f, "%zu\n", nrow);
  fprintf(f, "---------\n");
  fprintf(f, "Ncol\n");
  fprintf(f, "%zu\n", ncol);
  fprintf(f, "---------\n");
  fprintf(f, "PolarCase\nbistatic\n");
  fprintf(f, "---------\n");
  fprintf(f, "PolarType\nfull");
  fclose(f);
}
