#include"util.h"
// util.cpp
//
pthread_mutex_t print_mtx;
pthread_attr_t pthread_attr; // specify threads joinable
pthread_mutex_t pthread_next_j_mtx; // work queue
size_t pthread_next_j; // next job to run
size_t pthread_start_j, pthread_end_j; // start and end indices for job
void (*pthread_eval)(size_t); // function pointer to execute in parallel, over range start_j:end_j inclusive

void init_mtx(){
  // mutex setup
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pthread_next_j_mtx, NULL);
}

void cprint(string s){
  pthread_mutex_lock(&print_mtx);
  cout << s << endl;
  pthread_mutex_unlock(&print_mtx);
}

void * worker_fun(void * arg){
  size_t k, my_next_j;
  k = (size_t)arg;
  // cprint(str("worker_fun(") + std::to_string(k) + str(")"));

  while(1){
    // try to pick up a job
    mtx_lock(&pthread_next_j_mtx);
    my_next_j = pthread_next_j ++; // index of data this thread should pick up if it can
    mtx_unlock(&pthread_next_j_mtx);

    if(my_next_j >= pthread_end_j){
      cprint(str("\texit thread ") + to_string(k));
      return(NULL);
    }
    pthread_eval(my_next_j); // perform action segment
  }
}

void parfor(size_t start_j, size_t end_j, void(*eval)(size_t)){
  pthread_eval = eval; // set global function pointer
  pthread_start_j = start_j;
  pthread_end_j = end_j;

  pthread_next_j = start_j; // pthread_next_j_mtx is the lock on this variable
  size_t n_cores = sysconf(_SC_NPROCESSORS_ONLN);
  cout << "Number of cores: " << n_cores << endl;

  // allocate threads
  // // make the threads joinable
  pthread_attr_init(&pthread_attr);
  pthread_attr_setdetachstate(&pthread_attr, PTHREAD_CREATE_JOINABLE);
  pthread_t * my_pthread = new pthread_t[n_cores];
  size_t j;
  for0(j, n_cores){
    pthread_create(&my_pthread[j], &pthread_attr, worker_fun, (void *)j);
  }

  // wait for threads to finish
  for0(j, n_cores){
    pthread_join(my_pthread[j], NULL);
  }
  delete my_pthread;
}

void writeHeader(const char * filename, int NRows, int NCols, int NBand){
  time_t rawtime;
  struct tm * timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  FILE * f = fopen(filename, "w");
  if(!f){
    printf("Error: could not open file %s\n", filename);
    exit(1);
  }
  int datatype = 4;
  fprintf(f, "ENVI\n");
  fprintf(f, "description = {%s}\n",strip(string(asctime(timeinfo))).c_str());
  fprintf(f, "samples = %d\n", NCols);
  fprintf(f, "lines = %d\n", NRows);
  fprintf(f, "bands = %d\n", NBand);
  fprintf(f, "header offset = 0\n");
  fprintf(f, "file type = ENVI Standard\n");
  fprintf(f, "data type = %d\n",datatype);
  fprintf(f, "interleave = bsq\n");
  fprintf(f, "sensor type = Unknown\n");
  fprintf(f, "byte order = 0\n");
  fprintf(f, "wavelength units = Unknown\n");
  fclose(f);
  printf("w %s\n",filename);
}

vector<string> parseHeaderFile(string hfn, size_t & NRow, size_t & NCol, size_t & NBand){
  vector<string> bandNames;
  if(!exists(hfn)){
    printf("%sError: couldn't find header file\n", KRED);
  }
  else{
    vector<string> lines = readLines(hfn);
    vector<string>::iterator it;
    for(it=lines.begin(); it!=lines.end(); it++){
      string sss(*it);
      vector<string> splitLine(split(sss, '='));
      if(splitLine.size()==2){
        if(strncmp(strip(splitLine[0]).c_str(), "samples", 7) == 0){
          NCol = atoi(strip(splitLine[1]).c_str());
        }
        if(strncmp(strip(splitLine[0]).c_str(), "lines", 5) == 0){
          NRow = atoi(strip(splitLine[1]).c_str());
        }
        if(strncmp(strip(splitLine[0]).c_str(), "bands", 5)== 0){
          NBand = atoi(strip(splitLine[1]).c_str());
        }
        if(strncmp(strip(splitLine[0]).c_str(), "band names", 10) == 0){
          string bandNameList(trim(trim(strip(splitLine[1]),'{'),'}'));
          bandNames = split(bandNameList, ',');
        }
      }
    }
  }
  return bandNames;
}

/* split a string (a-la python) */
vector<string> split(string s, char delim){
  vector<string> ret;
  long int N = s.size();
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

vector<string> split(string s){
  return split(s, ' ');
}

vector<string> readLines(string fn){
  vector<string> ret;
  long int fs = getFileSize(fn);
  char * fd = (char *)(void *)malloc(fs);
  memset(fd, '\0',fs);
  FILE * f = fopen(fn.c_str(), "rb");
  long int br = fread(fd, fs, 1, f);
  fclose(f);
  string s(fd);
  free(fd);
  ret = split(s, '\n');
  return(ret);
}

string getHeaderFileName( string fn){
  string gfn(trim(fn, '\"'));
  string hfn(gfn + string(".hdr"));
  string hfn2((gfn.substr(0, gfn.size()-3)) + string("hdr"));
  if(exists(hfn)){
    return hfn;
  }
  else if(exists(hfn2)){
    return hfn2;
  }
  else{
    printf("%sError: could not find header file [%s] or [%s]\n", KRED, hfn.c_str(), hfn2.c_str());
    return string("");
  }
}

long int getFileSize(std::string fn){
  ifstream i;
  i.open(fn.c_str(), ios::binary);
  if(!i.is_open()){
    cout << "error: couldn't open file: " << fn << endl;
    return -1;
  }
  i.seekg(0, ios::end);
  long int len = i.tellg();
  return(len);
}

bool exists(string fn){
  if(getFileSize(fn) > 0){
    printf("%sFound file %s%s\n%s", KGRN, KRED, fn.c_str(), KNRM);
    return true;
  }
  return false;
}

/* special case of split (for newline character) */
vector<string> split(string s, char delim, long int i){
  //delimiter unused-- function unfinished. need to test this function properly
  vector<string> ret;
  long int N = s.size();
  if(N == 0) return ret;
  istringstream iss(s);
  string token;
  while(getline(iss, token)){
    ret.push_back(token);
  }
  return ret;
}

/*strip leading or trailing whitespace from a string*/
string strip(string s){
  string ret("");
  long int i, j, N;
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

/*trim leading or trailing characters from a string*/
string trim(string s, char a){
  string ret("");
  long int i, j, N;
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

vector<string> parse_band_names(string fn){
  if(!exists(fn)) err("parse_band_names: header file not found");
 
  vector<string> lines(readLines(fn)); // read header file
  for(vector<string>::iterator it = lines.begin(); it != lines.end(); it++){
    vector<string> w(split(*it));
  }


}




// parameters for (full-res) image subset extraction
size_t load_sub_np;
size_t load_sub_nb;
size_t load_sub_mm;
size_t load_sub_i_start;
size_t load_sub_j_start;
size_t load_sub_nc;
float * load_sub_dat3;
string load_sub_infile;

void load_sub(size_t k){
  // load one band of a rectangular image subset
  float d;
  FILE * f = fopen(load_sub_infile.c_str(), "rb");
  // printf("band %zu of %zu..\n", k + 1, load_sub_nb);
  size_t ki = k * load_sub_np;
  size_t kmi = k * load_sub_mm * load_sub_mm;
  for(size_t i = load_sub_i_start; i < load_sub_mm + load_sub_i_start; i++){
    size_t j = load_sub_j_start;
    size_t p = ki + (i * load_sub_nc) + j;
    p *= sizeof(float); // byte pos'n in file
    size_t jp = kmi + ((i - load_sub_i_start) * load_sub_mm); // + (j - j_start);

    // read row
    fseek(f, p, SEEK_SET);
    fread(&load_sub_dat3[jp], load_sub_mm, sizeof(float), f);
  }
  fclose(f);
}
