#include"misc.h"

string cwd(){
  char s[PATH_MAX];
  _cwd(s, PATH_MAX);
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
  if(!f) err("failed to open file for writing");
  return f;
}

/* get size of file pointer */
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

void rewind(ifstream &a){
  a.clear();
  a.seekg(0);
}

bool operator < (const f_idx& a, const f_idx&b){
  return a.d > b.d; // priority_queue max first: we want min first
}

// read header file
void hread(str hfn, size_t & nrow, size_t & ncol, size_t & nband){
  str line;
  vector<str> words;
  ifstream hf(hfn);
  nrow = ncol = nband = 0;
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
    }
  }
  cout << "hread: " << hfn << " nrow: " << nrow << " ncol: " << ncol << " nband: " << nband << endl;
  hf.close();
}

void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband){
  cout << "+w " << hfn << endl;  
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
  hf.close();
}

str hdr_fn(str fn, bool create=false){
  str hfn(fn + str(".hdr"));
  if(exists(hfn)) return hfn;
  int s = fn.size();
  str b(fn);
  b = b.substr(0, s - 4);
  str hfn2(b + str(".hdr"));
  if(exists(hfn2)) return hfn2;
  if(create){
    return hfn;
  }
  else{
    err(str("could not find header at: ") + hfn + str(" or at: ") + hfn2);
  }
  return hfn;
}

str hdr_fn(str fn){
  return hdr_fn(fn, false);
}


float * falloc(size_t nf){
  return (float *) alloc(nf * (size_t)sizeof(float));
}

// read binary file
float * bread(str bfn, size_t nrow, size_t ncol, size_t nband){
  cout << "bread(" << bfn << ", " << nrow << ", " << ncol << ", " << nband << ")" << endl;
  FILE * f = fopen(bfn.c_str(), "rb");
  size_t nf = nrow * ncol * nband;
  float * dat = falloc(nf);
  size_t nr = fread(dat, nf * (size_t)sizeof(float), 1, f);
  printf("nr %zu\n", nr);
  //if(nr != 1) err("failed to read data");
  fclose(f);
  return dat;
}

pthread_mutex_t print_mutex;

void cprint(str s){
  pthread_mutex_lock(&print_mutex);
  cout << s << endl;
  pthread_mutex_unlock(&print_mutex);
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


vector<string> parseHeaderFile(string hfn, size_t & NRow, size_t & NCol, size_t & NBand){
  vector<string> bandNames;
  if(!exists(hfn)){
    printf("Error: couldn't find header file\n");
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





