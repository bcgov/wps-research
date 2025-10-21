#ifndef _MISC_H_
#define _MISC_H_
//---------------------------------------------------------------------------//
// misc.h: helper functions //
// date: 20190124 //
//---------------------------------------------------------------------------//

/* shorthand for for loops from 0 to N */
#define for0(i,n) for(i = 0; i < n; i++)

#define STR_MAX 16384

#include<map>
#include<set>
#include<list>
#include<cmath>
#include<queue>
#include<cfloat>
#include<vector>
#include<string>
#include<sstream>
#include<fstream>
#include<iostream>
#include<limits.h>
#include<memory.h>
#include<pthread.h>
#include<algorithm>
#include"ansicolor.h"
#include<unordered_map>
#include<cctype>
#include<locale>
#include<stack>
#include<unordered_set>
#include<unordered_map>
#include<stdexcept>
#include<sched.h>

using namespace std;

inline char sep(){
  #ifdef _WIN32
  return '\\';
  #else
  return '/';
  #endif
}

inline string psep(){
  char s[2];
  s[0] = sep();
  s[1] = '\0';
  return string(s);
}


#include <stdio.h> /* defines FILENAME_MAX */
#ifdef WINDOWS
#include <direct.h>
#define _cwd _getcwd
#else
#include <unistd.h>
#define _cwd getcwd
#endif

std::string exec(const char* cmd);

void rewind(ifstream &a);

#define str string
string cwd();

/* split a string (a-la python) */
vector<string> split(string s, char delim);
vector<string> split(string s); // comma
vector<string> split_special(string s); // comma with possible commas inside quotation marks!
string join(const char * delim, vector<string> s);

// output operator for a vector container
template<class T> std::ostream& operator << (std::ostream& os, const std::vector<T>& v){
  os << "[";
  for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << " '" << *ii << "'";
  }
  os << "]";
  return os;
}

// output operator for set container
template<class T> std::ostream& operator << (std::ostream& os, const std::set<T>& v){
  os << "{";
  for (typename std::set<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << " " << *ii;
  }
  os << "}";
  return os;
}

// output operator for map container
template<class A, class B> std::ostream& operator << (std::ostream& os, const std::map<A, B>& v){
  os << "{";
  for (typename std::map<A, B>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << ii->first << ":" << ii->second;
    if(std::next(ii) != v.end()){
      os << ",\n"; //endl;
    }
  }
  os << "}";
  return os;
}

// output operator for uo_set container
template<class T> std::ostream& operator << (std::ostream& os, const std::unordered_set<T>& v){
  os << "{";
  for (typename std::unordered_set<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << " " << *ii;
  }
  os << "}";
  return os;
}

// output operator for uo_map container
template<class A, class B> std::ostream& operator << (std::ostream& os, const std::unordered_map<A, B>& v){
  os << "{";
  for (typename std::unordered_map<A, B>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << ii->first << ":" << ii->second;
    if(std::next(ii) != v.end()){
      os << ",\n"; //endl;
    }
  }
  os << "}";
  return os;
}


// output operator for list container
template<class T> std::ostream& operator << (std::ostream& os, const std::list<T>& v){
  os << "{";
  for (typename std::list<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << " " << *ii;
  }
  os << "}";
  return os;
}

void err(string msg);

void err(const char * msg);

/* allocate memory */
inline void * alloc(size_t nb){
  void * d = malloc(nb);
  if(!d){
    printf("%zu\n", nb);
    err("failed to allocate memory");
  }
  memset(d, '\0', nb);
  return d;
}

inline char * calloc(size_t nb){
  char * d = (char *)malloc(nb);
  if(!d){
    printf("%zu\n", nb);
    err("failed to allocate memory");
  }
  memset(d, '\0', nb);
  return d;
}

inline int * ialloc(size_t ni){
  int * d = (int *)malloc(ni * sizeof(int));
  if(!d){
    printf("%zu\n", ni);
    err("failed to allocate memory");
  }
  memset(d, '\0', ni * sizeof(int));
  return d;
}

inline size_t * stalloc(size_t ni){
  size_t * d = (size_t *)malloc(ni * sizeof(size_t));
  if(!d){
    printf("%zu\n", ni);
    err("failed to allocate memory");
  }
  memset(d, '\0', ni * sizeof(size_t));
  return d;
}


//a trim from start (in place)
/*bool is_space(int ch){
  return !std::isspace(ch);
}*/

bool not_space(int data); 
/* 
   return !std::isspace(data);
}
*/

static inline void ltrim(std::string &s){
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), /* [](int ch){
    return !std::isspace(ch);
  }*/ 
  not_space
  ));
}

// trim from end (in place)
static inline void rtrim(std::string &s){
  s.erase(std::find_if(s.rbegin(), s.rend(),
 /* [](int ch){
    return !std::isspace(ch);
  }*/
 not_space
  ).base(), s.end());
}

static inline void ltrim(str & s, str chars){
  s.erase(0, s.find_first_not_of(chars));
}

static inline void rtrim(str & s, str chars){
  std::size_t found = s.find_last_not_of(chars);
  // printf("found %zu strlen %zu\n", found, (size_t)strlen(s.c_str()));
  if (found!=std::string::npos) s.erase(found + 1);

}

static inline void trim(str & s, str chars){
  ltrim(s, chars);
  rtrim(s, chars);
}

// trim from both ends (in place)
static inline void trim(std::string &s){
  ltrim(s);
  rtrim(s);
}

// trim from start (copying): not implemented properly
static inline std::string ltrim_copy(std::string s){
  ltrim(s);
  return s;
}

// trim from end (copying): not implemented properly
static inline std::string rtrim_copy(std::string s){
  rtrim(s);
  return s;
}

// trim from both ends (copying): not implemented properly
static inline std::string trim_copy(std::string s){
  trim(s);
  return s;
}

static inline void trim(std::string &s, char delim){
  str ret("");
  int end = s.size() - 1;
  int start = 0;
  while(s[start] == delim) start += 1;
  while(s[end] == delim) end -= 1;
  s = s.substr(start, 1 + end - start);
}

#define strip trim

/* convert to lower case */
static void inline lower(std::string & s){
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
}

static inline std::string lower_copy(std::string &s){
  string r(s);
  std::transform(r.begin(), r.end(), r.begin(), ::tolower);
  return r;
}

/* get size of file pointer */
size_t size(FILE * f);
size_t fsize(string fn);

// in-memory reader (writer to be implemented)
// note this should be able to modulate between available protocols (like ifstream, ofstream, etc. , fwrite, fread, if available)

FILE * wopen(string fn);
FILE * wopen(const char * fn);
FILE * ropen(string fn);
FILE * ropen(const char * fn);

class f_idx{
  public: // float, index tuple object
  float d;
  size_t idx;
  f_idx(float d_ = 0., size_t idx_ = 0){
    d = d_;
    idx = idx_;
  }
  f_idx(const f_idx &a){
    d = a.d;
    idx = a.idx;
  }
};

bool operator<(const f_idx& a, const f_idx&b);

// read header file
size_t hread(str hfn, size_t & nrow, size_t & ncol, size_t & nband);
size_t hread(str hfn, size_t & nrow, size_t & ncol, size_t & nband, vector<string>& bandNames);

void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband);
void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband, size_t data_type);
void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband, size_t data_type, vector<string> & bandNames);

// append class_names field to ENVI header file
void happend_class_names(const string& hfn, const vector<string>& class_names);

float * falloc(size_t nf);
double * dalloc(size_t nd);

char * read(str bfn, size_t & n_bytes); // read binary file in char format 
float * bread(str bfn, size_t nrow, size_t ncol, size_t nband); // read binary file  ( assumed float)
void bwrite(float * d, str bfn, size_t nrow, size_t ncol, size_t nband); // write binary file
void bappend(float * d, FILE * f, size_t n_float); //, bool start=false); // append (e.g. a band) to a binary file

int hsv_to_rgb(float *r, float *g, float *b, float h, float s, float v);
int rgb_to_hsv(float r, float g, float b, float * h, float * s, float * v);


str hdr_fn(str fn); //create = false
str hdr_fn(str fn, bool create);

/* get size of file pointer */
size_t size(FILE * f);
size_t fsize(string fn);
size_t fsize(const char * fn);
bool exists(str fn);

float * load_envi(str in_f, size_t & nrow, size_t & ncol, size_t & nband);

// need to compare / merge this with f_idx above?
class f_i{
  public: // float, index tuple object
  float d;
  size_t i;
  f_i(float d_ = 0., size_t i_ = 0){
    d = d_;
    i = i_;
  }
  f_i(const f_i &a){
    d = a.d;
    i = a.i;
  }
};

class f_ij{
  public: // float, two-index tuple object
  float d;
  size_t i, j;
  f_ij(float d_ = 0., size_t i_ = 0, size_t j_ = 0){
    d = d_;
    i = i_;
    j = j_;
  }
  f_ij(const f_ij &a){
    d = a.d;
    i = a.i;
    j = a.j;
  }
};

// 20221028 now we know we need f_i<template class> definition!
class f_v{
  public:
    size_t d;
    vector<float> v;
  f_v(size_t dd, vector<float> vv){
    d = dd;
    v = vv;
  }
  f_v(const f_v &a){
    d = a.d;
    v = a.v;
  }
};


bool operator<(const f_i& a, const f_i&b);
bool operator<(const f_ij& a, const f_ij&b);
bool operator<(const f_v& a, const f_v&b);

#define mtx_lock pthread_mutex_lock
#define mtx_unlock pthread_mutex_unlock

// zero pad a string (from left)
str zero_pad(str x, int n_zero);

// input / output stuff
void int_write(size_t value, str fn);
size_t int_read(str fn);
size_t * ints_read(str fn); // read a number of ints from files
float * float_read(str fn); // read floats from a file
float * float_read(str fn, size_t &n);
void float_write(float * d, size_t n, str fn);


// parallelism stuff
extern pthread_mutex_t print_mtx;
void cprint(str s);

// parfor stuff
void set_thread_affinity(pthread_t thread, int cpuIndex);
extern pthread_attr_t pt_attr; // specify threads joinable
extern pthread_mutex_t pt_next_j_mtx;
extern size_t pt_nxt_j;
extern size_t pt_start_j;
extern size_t pt_end_j;
void init_mtx();
void * pt_worker_fun(void * arg); // worker function
extern void (*pt_eval)(size_t); // function pointer to execute in parallel, over range start_j:end_j inclusive
void parfor(size_t start_j, size_t end_j, void(*eval)(size_t), int cores_use);
void parfor(size_t start_j, size_t end_j, void(*eval)(size_t));

vector< vector<str> > read_csv(str fn, vector<str> & hdr);

vector<string> parseHeaderFile(string hfn, size_t & NRow, size_t & NCol, size_t & NBand);


vector<string> readLines(string fn); // vector<string> split(string s, char delim);
// need to replace string with SA concept, to make this work! Use "binary string" instead of string
vector<string> split(char * s, size_t s_len, char delim);


/*trim leading or trailing characters from a string*/
string trim2(string s, char a);

/*convert char to string: single character: interpret whitspace as space character */
string chartos(char s);

/*strip leading or trailing whitespace from a string*/
string strip_space(string s);
vector<string> parse_band_names(string fn);
bool contains(string s1, string s2);

void status(size_t i, size_t of);

void write_config(str fn, size_t nrow, size_t ncol);
int run(str s);

void rgb2cmyk(float R, float G, float B, float * Cyan, float* Magenta, float* Yellow, float* Black);

#endif
