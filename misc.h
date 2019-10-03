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
#include<cmath>
#include<queue>
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


using namespace std;

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

template<class T> std::ostream& operator << (std::ostream& os, const std::vector<T>& v){
  os << "[";
  for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << " '" << *ii << "'";
  }
  os << "]";
  return os;
}

template<class T> std::ostream& operator << (std::ostream& os, const std::set<T>& v){
  os << "{";
  for (typename std::set<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << " " << *ii;
  }
  os << "}";
  return os;
}

template<class A, class B> std::ostream& operator << (std::ostream& os, const std::map<A, B>& v){
  os << "{" << endl;
  for (typename std::map<A, B>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << ii->first << ":" << ii->second << ","; //endl;
  }
  os << "}" << endl;
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

class f_idx{
  public: // float, index tuple object
  float d;
  unsigned int idx;
  f_idx(float d_ = 0., unsigned int idx_ = 0){
    d = d_;
    idx = idx_;
  }
  f_idx(const f_idx &a){
    d = a.d;
    idx = a.idx;
  }
};

bool operator<(const f_idx& a, const f_idx&b);
//return a.d > b.d; // priority_queue max first: we want min first
//}

// read header file
void hread(str hfn, int & nrow, int & ncol, int & nband);
/*
      	str line;
  vector<str> words;
  ifstream hf(hfn);
  nrow = ncol = nband = 0;
  if(!hf.is_open()) err(str("failed to open header file: ") + hfn);
  while(getline(hf, line)){
    words = split(line, '=');
    if(words.size() == 2){
      strip(words[0]);
      str w(words[0]);
      int n = atoi(words[1].c_str());
      if(w == str("samples")) ncol = n;
      if(w == str("lines")) nrow = n;
      if(w == str("bands")) nband = n;
    }
  }
  hf.close();
}
*/

float * falloc(size_t nf);
//
//      	return (float *) alloc(nf * (size_t)sizeof(float));
//}

// read binary file
float * bread(str bfn, int nrow, int ncol, int nband);
/*
       	FILE * f = fopen(bfn.c_str(), "rb");
  size_t nf = (size_t)nrow * (size_t)ncol * (size_t)nband;
  float * dat = falloc(nf);
  size_t nr = fread(dat, nf * (size_t)sizeof(float), 1, f);
  if(nr != 1) err("failed to read data");
  fclose(f);
  return dat;
}
*/

extern  pthread_mutex_t print_mutex;


void cprint(str s);
/*
      	pthread_mutex_lock(&print_mutex);
  cout << s << endl;
  pthread_mutex_unlock(&print_mutex);
}
*/


#endif
