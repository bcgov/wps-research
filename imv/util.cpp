#include"util.h"
// util.cpp
//
pthread_mutex_t print_mtx;
pthread_attr_t pthread_attr; // specify threads joinable
pthread_mutex_t pthread_next_j_mtx; // work queue
size_t pthread_next_j; // next job to run
size_t pthread_start_j, pthread_end_j; // start and end indices for job
void (*pthread_eval)(size_t); // function pointer to execute in parallel, over range start_j:end_j inclusive

void err(char * msg){
  printf("Error: %s\n", msg);
  exit(1);
}

void err(string msg){
  err(msg.c_str());
}

// if it is an int, should be able to cast to int and back!
bool is_int(string i){
  int x = atoi(i.c_str());
  return (std::to_string(x) == i);
}

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
      // cprint(str("\texit thread ") + to_string(k));

      return(NULL);
    }
    pthread_eval(my_next_j); // perform action segment
  }
}

void parfor(size_t start_j, size_t end_j, void(*eval)(size_t), size_t cores_use){
  // ideally the worker fun would be an inline (inside of here)

  pthread_eval = eval; // set global function pointer
  pthread_start_j = start_j;
  pthread_end_j = end_j;

  pthread_next_j = start_j; // pthread_next_j_mtx is the lock on this variable
  size_t n_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if(cores_use > 0) n_cores = cores_use;
  cout << "Using " << n_cores << " threads.." << endl;

  // allocate threads, make threads joinable whatever that means
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
  // delete my_pthread;
  cprint(str("return parfor()"));
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
  fprintf(f, "band names = {band 1");
  for(int i = 1; i < NBand; i++){
    fprintf(f, ",\nband %d", i);
  }
  fprintf(f, "}\n");
  fclose(f);
  printf("w %s\n",filename);
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

vector<string> split(string s){
  return split(s, ' ');
}

vector<string> readLines(string fn){
  vector<string> ret;
  size_t fs = getFileSize(fn);
  char * fd = (char *)(void *)malloc(fs);
  memset(fd, '\0',fs);
  FILE * f = fopen(fn.c_str(), "rb");
  size_t br = fread(fd, fs, 1, f);
  fclose(f);
  // free(fd);
  ret = split(fd, fs, '\n');
  return(ret);
}

string getHeaderFileName( string fn){
  string gfn(trim(fn, '\"'));
  string hfn(gfn + string(".hdr"));
  string hfn2((gfn.substr(0, gfn.size()-3)) + string("hdr"));
  if(exists(hfn)) return hfn;
  if(exists(hfn2)) return hfn2;
  printf("%sError: could not find header file [%s] or [%s]\n", KRED, hfn.c_str(), hfn2.c_str());
  err("");
}

size_t getFileSize(std::string fn){
  ifstream i;
  i.open(fn.c_str(), ios::binary);
  bool condition = i.is_open();
  if(!condition){
    // cout << "Warning: couldn't open file: " << fn << endl;
    return 0;
  }
  i.seekg(0, ios::end);
  size_t len = i.tellg();
  return(len);
}

bool exists(string fn){
  size_t fs = getFileSize(fn);
  bool condition = fs > 0;
  if(condition){
    // printf("%sFound file %s%s\n%s", KGRN, KRED, fn.c_str(), KNRM);
  }
  else{
    printf("%sError: failed to find file %s%s\n%s", KGRN, KRED, fn.c_str(), KNRM);
  }

  return condition;
}

/* special case of split (for newline character) */
vector<string> split(string s, char delim, long int i){
  //delimiter unused-- function unfinished. need to test this function properly
  vector<string> ret;
  size_t N = s.size();
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

/*trim leading or trailing characters from a string*/
string trim(string s, char a){
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

vector<string> parse_band_names(string fn){
  if(!exists(fn)) err("parse_band_names: header file not found");

  str band("band");
  str names("names");
  size_t ci = 0;
  size_t bni = -1; // band names start index
  vector<string> lines(readLines(fn)); // read header file
  for(vector<string>::iterator it = lines.begin(); it != lines.end(); it++){
    // cout << "\t" << *it << endl;
    vector<string> w(split(*it));
    //for(vector<string>::iterator it2 = w.begin(); it2 != w.end(); it2++){
      // cout << "\t\t" << *it2 << endl;
    //}
    if(w.size() >= 2){
      if((w[0] == band) && (w[1] == names)){
        bni = ci;
        break;
      }
    }
    ci ++;
  }

  vector<string> band_names;

  // parse first band name
  vector<string> w(split(lines[bni], '{'));
  string bn(w[1]);
  bn = trim(bn, ',');
  //cout << bn << endl;
  band_names.push_back(bn);

  // parse middle band names
  for(ci = bni + 1; ci < lines.size() - 1; ci++){
    str line(lines[ci]);
    line = trim(line, ',');
    //cout << line << endl;
    band_names.push_back(line);
  }

  // parse last band name
  if(true){
    str w(lines[lines.size() -1]);
    w = trim(w, '}');
    band_names.push_back(w);
  }
  return band_names;
}

vector<int> parse_groundref_names(string fn, int n_groundref){
  // groundref one-hot encoded bands, are assumed to NOT contain numbers within the band name
  int ci = 0;
  int debug = false; // turn this on to debug problems with ground-ref bandnames recognition
  int at_gt = false;
  vector<int> results;
  vector<string> names(parse_band_names(fn));
  if(debug) cout << "PARSE_GROUNDREF_NAMES" << endl; // assume a groundref name doesn't have numbers in it!

  int guess_groundref = false;
  if(guess_groundref){

    for(vector<string>::iterator it = names.begin(); it != names.end(); it++){
      string x(*it);
      std::replace(x.begin(), x.end(), '_', ' ');
      vector<string> w(split(x));
      bool has_number = false;

      for(vector<string>::iterator it2 = w.begin(); it2 != w.end(); it2++){
        string y(*it2);
        y = strip_leading_zeros(y);
        y = trim(y, '(');
        y = trim(y, ')');
        if(debug) cout << "\t\t" << y << endl;
        if(is_int(y)){
          if(debug) cout << "\t\t\tNUMBER" << endl;
          has_number = true;
          break;
        }
      }
      if(!has_number || at_gt){
        if(debug) cout << "\tgroundref: " << *it << " bi " << ci + 1 << "\n\t\tbi (0-indexed) " << ci << endl;
        results.push_back(ci);
        at_gt = true;
      }
      ci ++;
    }
    if(debug) cout << "Exit loop" << endl;
  }
  else{
    int start = names.size() - n_groundref;
    for(ci = start; ci < names.size(); ci ++){
      printf("groundref[%s]\n", names[ci].c_str());
      results.push_back(ci);
    }
  }
  return results;

}

int hsv_to_rgb(float *r, float *g, float *b, float h, float s, float v){
  if(h > 360. || h < 0.){
    printf("H: HSV out of range %f %f %f\n", h, s, v);
    return(1);
  }
  if(s < 0. || s > 1.){
    printf("S: HSV out of range %f %f %f\n", h, s, v);
    return(1);
  }
  if(v < 0. || v > 1.){
    printf("V: HSV out of range %f %f %f\n", h, s, v);
    return(1);
  }
  if(h == 360.){
    h = 0;
  }
  int i;
  float f, p, q, t;
  if(s == 0) {
    *r = *g = *b = v; // achromatic (grey)
    return 0;
  }
  float H,S,V;
  H = h; V = v; S = s;
  h /= 60.; // sector 0 to 5
  i = (int)floor(h);
  f = h - i; // factorial part of h
  p = v * (1. - s);
  q = v * (1. - s * f);
  t = v * (1. - s * (1 - f));
  switch(i){
    case 0: *r = v; *g = t; *b = p; break;
    case 1: *r = q; *g = v; *b = p; break;
    case 2: *r = p; *g = v; *b = t; break;
    case 3: *r = p; *g = q; *b = v; break;
    case 4: *r = t; *g = p; *b = v; break;
    case 5: *r = v; *g = p; *b = q; break;
    default:
    printf("\nError: hsv2rgb: i=%d hsv= %f %f %f\n", i, H, S, V);
  }
  return 0;
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
SA<size_t> * load_sub_i;
SA<size_t> * load_sub_j;

void load_sub(size_t k){
  float d;
  FILE * f = fopen(load_sub_infile.c_str(), "rb");
  // printf("load band %zu of %zu on rect. image subset..\n", k + 1, load_sub_nb);
  size_t ki = k * load_sub_np;
  size_t kmi = k * load_sub_mm * load_sub_mm;

  for(size_t i = load_sub_i_start; i < load_sub_mm + load_sub_i_start; i++){
    size_t j = load_sub_j_start;
    size_t jp = kmi + ((i - load_sub_i_start) * load_sub_mm);
    size_t p = (ki + (i * load_sub_nc) + j) * sizeof(float); // file byte pos

    fseek(f, p, SEEK_SET);
    size_t nr = fread(&load_sub_dat3[jp], load_sub_mm, sizeof(float), f); // read row

    if(k == 0){
      size_t mi, mj, jj;
      size_t * lsj, *lsi; // record transformation between global img coordinates, and subimage
      lsj = &load_sub_j->at(0);
      lsi = &load_sub_i->at(0);
      mi = (i - load_sub_i_start) * load_sub_mm;
      for(jj = load_sub_j_start; jj < load_sub_mm + load_sub_j_start; jj++){
        mj = (jj - load_sub_j_start);
        // printf("mi %zu j mj %zu i %zu j %zu\n", mi, mj, i, jj); // only need to do this for one band
        lsi[mi + mj] = i;
        lsj[mi + mj] = jj;
      }
    }
  }
  fclose(f);
}

// parameters for (full-res) image subset nan-application
size_t nan_sub_np;
size_t nan_sub_nb;
size_t nan_sub_mm;
size_t nan_sub_i_start;
size_t nan_sub_j_start;
size_t nan_sub_nc;
float * nan_sub_dat3;
string nan_sub_infile;
SA<size_t> * nan_sub_i;
SA<size_t> * nan_sub_j;

void nan_sub(size_t k){
  float d;
  FILE * f = fopen(nan_sub_infile.c_str(), "r+b");
  // printf("nan band %zu of %zu on rect. image subset..\n", k + 1, load_sub_nb);
  size_t ki = k * nan_sub_np;
  size_t kmi = k * nan_sub_mm * load_sub_mm;

  float * w_nan = (float *) alloc(sizeof(float) * load_sub_mm);
  for(size_t i = 0; i < load_sub_mm; i++){
    w_nan[i] = NAN;
  }

  for(size_t i = nan_sub_i_start; i < load_sub_mm + load_sub_i_start; i++){
    size_t j = nan_sub_j_start;
    size_t jp = kmi + ((i - nan_sub_i_start) * load_sub_mm);
    size_t p = (ki + (i * nan_sub_nc) + j) * sizeof(float); // file byte pos

    fseek(f, p, SEEK_SET);
    // size_t nr = fread(&nan_sub_dat3[jp], load_sub_mm, sizeof(float), f); // read row
    size_t nr = fwrite(w_nan, load_sub_mm, sizeof(float), f);

    if(k == 0){
      size_t mi, mj, jj;
      size_t * lsj, *lsi; // record transformation between global img coordinates, and subimage
      lsj = &nan_sub_j->at(0);
      lsi = &nan_sub_i->at(0);
      mi = (i - nan_sub_i_start) * load_sub_mm;
      for(jj = nan_sub_j_start; jj < load_sub_mm + load_sub_j_start; jj++){
        mj = (jj - nan_sub_j_start);
        // printf("mi %zu j mj %zu i %zu j %zu\n", mi, mj, i, jj); // only need to do this for one band
        lsi[mi + mj] = i;
        lsj[mi + mj] = jj;
      }
    }
  }
  fclose(f);
  free(w_nan);
}

void zero_sub(size_t k){
  float d;
  FILE * f = fopen(nan_sub_infile.c_str(), "r+b");
  // printf("nan band %zu of %zu on rect. image subset..\n", k + 1, load_sub_nb);
  size_t ki = k * nan_sub_np;
  size_t kmi = k * nan_sub_mm * load_sub_mm;

  float * w_nan = (float *) alloc(sizeof(float) * load_sub_mm);
  for(size_t i = 0; i < load_sub_mm; i++){
    w_nan[i] = 0.; //NAN;
  }

  for(size_t i = nan_sub_i_start; i < load_sub_mm + load_sub_i_start; i++){
    size_t j = nan_sub_j_start;
    size_t jp = kmi + ((i - nan_sub_i_start) * load_sub_mm);
    size_t p = (ki + (i * nan_sub_nc) + j) * sizeof(float); // file byte pos

    fseek(f, p, SEEK_SET);
    // size_t nr = fread(&nan_sub_dat3[jp], load_sub_mm, sizeof(float), f); // read row
    size_t nr = fwrite(w_nan, load_sub_mm, sizeof(float), f);

    if(k == 0){
      size_t mi, mj, jj;
      size_t * lsj, *lsi; // record transformation between global img coordinates, and subimage
      lsj = &nan_sub_j->at(0);
      lsi = &nan_sub_i->at(0);
      mi = (i - nan_sub_i_start) * load_sub_mm;
      for(jj = nan_sub_j_start; jj < load_sub_mm + load_sub_j_start; jj++){
        mj = (jj - nan_sub_j_start);
        // printf("mi %zu j mj %zu i %zu j %zu\n", mi, mj, i, jj); // only need to do this for one band
        lsi[mi + mj] = i;
        lsj[mi + mj] = jj;
      }
    }
  }
  fclose(f);
  free(w_nan);
}





// scene subsampling, parallelized by band
size_t mlk_scene_nb; // infile nbands
size_t mlk_scene_nr; // infile nrows
size_t mlk_scene_nc; // infile ncols
size_t mlk_scene_nr2; // infile nrows
size_t mlk_scene_nc2; // infile ncols
size_t mlk_scene_np2;
float mlk_scene_scalef; // scaling factor
string * mlk_scene_fn; // input filename
float * mlk_scene_dat; // float data output
vector<int> * mlk_scene_groundref; // groundref indices

void multilook_scene(size_t k){
  // due a crude sampling of scene, for scene overview window
  size_t np = mlk_scene_nr * mlk_scene_nc;
  printf("Nrow %zu Ncol %zu\n", mlk_scene_nr, mlk_scene_nc);	
  
	float * bb = (float *)(void *)malloc(np * sizeof(float));
	if(!bb){
		err("failed to allocate memory");
	}
	else{
		memset(bb, '\0', np*sizeof(float));
	}

  set<int> groundref_set;
  for(vector<int>::iterator it = mlk_scene_groundref->begin(); it != mlk_scene_groundref->end(); it++){
    groundref_set.insert(*it);
  }

  // printf("scaling %d x %d image to %d x %d\n", nr, nc, nr2, nc2);
	printf("+w %s\n", mlk_scene_fn->c_str());
  FILE * f = fopen(mlk_scene_fn->c_str(), "rb");
	if(!f){
		err("failed to open input file");
  }

  printf("fread band %zu/%d\n", k + 1, mlk_scene_nb);
  size_t nbyte = np * sizeof(float);
	printf("SEEK %zu\n", nbyte * k);
  fseek(f, nbyte * k, SEEK_SET);

	size_t x;
	for0(x, mlk_scene_nr){
    if(x % 1000 == 0){
			printf("x=%zu/ %zu  SEEK %zu\n", x + 1, mlk_scene_nr, (np*k) + mlk_scene_nc * x);
		}
    fseek(f, (np * k) + mlk_scene_nc * x, SEEK_SET);
  	size_t nread = fread(&bb[mlk_scene_nc * x], sizeof(float), mlk_scene_nc, f); // read band
		if(nread != mlk_scene_nc){
    	printf("read %zu bytes instead of %zu (expected) from file: %s\n", nread, mlk_scene_nc * sizeof(float), mlk_scene_fn->c_str());
    	err("exit");
  	}
	}

  if(groundref_set.find(k) != groundref_set.end()){
    // assert all 1. / 0. for ground ref!
    for(size_t i = 0; i < np; i++){
      float d = bb[i];
      if(!(d == 0. || d == 1.)){
        printf("\tband_index %zu\n", k);
        err("assertion failed: that groundref be valued in {0,1} only");
      }
    }
  }

	printf("averaging..\n");
  for(size_t i = 0; i < mlk_scene_nr; i++){
    size_t ip = (size_t)floor(mlk_scene_scalef * (float)i);
    for(size_t j = 0; j < mlk_scene_nc; j++){
      size_t jp = (size_t)floor(mlk_scene_scalef * (float)j);
      mlk_scene_dat[k * mlk_scene_np2 + ip * mlk_scene_nc2 + jp] = bb[(i * mlk_scene_nc) + j];
      // last line could / should be += ?
    }
  }
	free(bb);
}

str strip_leading_zeros(str s){
  str ss(s);
  ss = ss.erase(0, min(ss.find_first_not_of('0'), ss.size() - 1));
  return(ss);
}

vector<vector<str>> read_csv(str fn, vector<str> & hdr){
  vector<vector<str>> output; // read lines from csv file
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

bool vin(vector<str> x, str a){
  // does a appear in x? should be templated
  vector<str>::iterator i;
  for(i = x.begin(); i != x.end(); i++) if(*i == a) return true;
  return false;
}

size_t vix(vector<str> x, str a){
  // does a appear in x? should be templated
  size_t ix = 0;
  vector<str>::iterator i;
  for(i = x.begin(); i != x.end(); i++){
    if(*i == a) return ix;
    ++ix;
  }
  err("should have checked for element before using vix");
}

size_t write_csv(str fn, vector<str> hdr, vector<vector<str>> lines){
  cout << "+w " << fn << endl;
  size_t n = hdr.size();
  FILE * f = fopen(fn.c_str(), "wb");
  if(!f) err("failed to open output file");
  cout << "hdr" << hdr << endl;
 
  size_t i, j; 
  cout << "hdr[0]: " << hdr[0] << endl;
  fprintf(f, "%s", hdr[0].c_str());
  for(i = 1; i < n; i++){
    cout << i << " " << hdr[i] << endl;
    fprintf(f, ",%s", hdr[i].c_str());
  }
  for0(i, lines.size()){
    if(lines[i].size() != n){
      while(lines[i].size() != n){
        lines[i].push_back(str(""));
      }
    }

    if(lines[i].size() != n){
      cout << "hdr: " << hdr << endl;
      cout << "n," << n << endl;
      cout << "lines[i].size() " << lines[i].size() << endl;
      cout << lines[i] << endl;
      err("warning: internal csv data formatting error");
    }
    str ij((lines[i])[0]);
    fprintf(f, "\n%s", ij.c_str()); 
    for(j = 1; j < lines[i].size(); j++){
        str ij((lines[i])[j]);
	fprintf(f, ",%s", ij.c_str()); 
    }
  }
  cout << "got here!" << endl;
  fclose(f);
  cout << "got here!" << endl;

  return 0;
}

std::string exec(const char* cmd){
  // run system command and catch result from stdout
  char buffer[16384]; // watch the limit, should have a growing-stack version of this
  std::string result = "";
  FILE* pipe = popen(cmd, "r");
  if (!pipe) throw std::runtime_error("popen() failed!");
  try{
    while(fgets(buffer, sizeof buffer, pipe) != NULL) result += buffer;
  }
  catch (...) {
    pclose(pipe);
    throw;
  }
  pclose(pipe);
  return result;
}
