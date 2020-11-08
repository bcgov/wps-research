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

#include <algorithm>
#include <cctype>
#include <locale>
#include<stack>

using namespace std;

inline char sep(){
  #ifdef _WIN32
  return '\\';
  #else
  return '/';
  #endif
}

#include <stdio.h> /* defines FILENAME_MAX */
#ifdef WINDOWS
#include <direct.h>
#define _cwd _getcwd
#else
#include <unistd.h>
#define _cwd getcwd
#endif

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

/*
float * falloc(size_t nf){
  return (float *) alloc(nf * sizeof(float));
}
*/

//a trim from start (in place)
static inline void ltrim(std::string &s){
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch){
    return !std::isspace(ch);
  }
  ));
}

// trim from end (in place)
static inline void rtrim(std::string &s){
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch){
    return !std::isspace(ch);
  }
  ).base(), s.end());
}

static inline void ltrim(str & s, str chars){
  s.erase(0, s.find_first_not_of(chars));
}

static inline void rtrim(str & s, str chars){
  std::size_t found = s.find_last_not_of(chars);
  printf("found %zu strlen %zu\n", found, (size_t)strlen(s.c_str()));
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
void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband);
void hwrite(str hfn, size_t nrow, size_t ncol, size_t nband, size_t data_type);

float * falloc(size_t nf);

float * bread(str bfn, size_t nrow, size_t ncol, size_t nband); // read binary file
void bwrite(float * d, str bfn, size_t nrow, size_t ncol, size_t nband); // write binary file

extern pthread_mutex_t print_mtx;
void cprint(str s);

int hsv_to_rgb(float *r, float *g, float *b, float h, float s, float v);

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

bool operator<(const f_i& a, const f_i&b);
bool operator<(const f_ij& a, const f_ij&b);

#define mtx_lock pthread_mutex_lock
#define mtx_unlock pthread_mutex_unlock
#endif

// zero pad a string (from left)
str zero_pad(str x, int n_zero);

void int_write(size_t value, str fn);
size_t int_read(str fn);

float * float_read(str fn, size_t &n);
void float_write(float * d, size_t n, str fn);