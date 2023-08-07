#pragma once
#ifndef UTIL_H
#define UTIL_H

#include"SA.h"
#include<set>
#include<map>
#include<math.h>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<memory.h>
#include<algorithm>
#include<iostream>
#include<pthread.h>
#include"ansicolor.h"

#include <stdio.h> /* defines FILENAME_MAX */
#ifdef WINDOWS
#include <direct.h>
#define _cwd _getcwd
#else
#include <unistd.h>
#define _cwd getcwd
#endif

inline char separator(){
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

/* shorthand for for loops from 0 to N */
#define for0(i,n) for(i = 0; i < n; i++)

using namespace std;
#define str string

void err(char * msg);
void err(string msg);
bool is_int(string i);

/*convert char to string: single character: interpret whitspace as space character */
string chartos(char s);

/*strip leading or trailing whitespace from a string*/
string strip(string s);

/*trim leading or trailing characters from a string*/
string trim(string s, char a);

size_t getFileSize(std::string fn);
bool exists(string fn);

/* special case of split (for newline character) */
vector<string> split(string s, char delim, long int i);

/* split a string (a-la python) */
vector<string> split(string s, char delim);

vector<string> split(string s);

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
  os << "\n{"; // endl;
  int ci = 0;
  for (typename std::map<A, B>::const_iterator ii = v.begin(); ii != v.end(); ++ii){
    os << (ci > 0 ? str("\n") : str("")) << ii->first << ":" << ii->second << ","; //endl;
    ci += 1;
  }
  os << "}" << endl;
  return os;
}

vector<string> readLines(string fn);
string getHeaderFileName( string fn);
vector<string> parseHeaderFile(string hfn, size_t & NRow, size_t & NCol, size_t & NBand);
void writeHeader(const char * filename, int NRows, int NCols, int NBand);

vector<string> parse_band_names(string fn); // read band names from header file
vector<int> parse_groundref_names(string fn, int n_groundref); // indices of groundref bands

str strip_leading_zeros(str s);


int hsv_to_rgb(float *r, float *g, float *b, float h, float s, float v);
// vector<float> color_wheel_rgb(size_t n_class);

// parallelism stuff
#define mtx_lock pthread_mutex_lock
#define mtx_unlock pthread_mutex_unlock

extern pthread_mutex_t print_mtx;
void cprint(str s);
extern pthread_attr_t pthread_attr; // specify threads joinable
extern pthread_mutex_t pthread_next_j_mtx;
extern size_t pthread_next_j;

extern size_t pthread_start_j;
extern size_t pthread_end_j;

void init_mtx();
void * worker_fun(void * arg); // worker function
extern void (*pthread_eval)(size_t); // function pointer to execute in parallel, over range start_j:end_j inclusive
void parfor(size_t start_j, size_t end_j, void(*eval)(size_t), size_t cores_use = 0);

// method for loading band of image, limited to subarea
extern size_t load_sub_np;
extern size_t load_sub_nb;
extern size_t load_sub_mm;
extern size_t load_sub_i_start;
extern size_t load_sub_j_start;
extern size_t load_sub_nc;
extern float * load_sub_dat3;
extern string load_sub_infile;
extern SA<size_t> * load_sub_i; // this is an array full of row indices for extracted stuff
extern SA<size_t> * load_sub_j; // this is an array full of col indices for extracted stuff
void load_sub(size_t k); // subset data loading, parallelized by band

// method for writing NaN over band of image, limited to subarea
extern size_t nan_sub_np;
extern size_t nan_sub_nb;
extern size_t nan_sub_mm;
extern size_t nan_sub_i_start;
extern size_t nan_sub_j_start;
extern size_t nan_sub_nc;
extern float * nan_sub_dat3;
extern string nan_sub_infile;
extern SA<size_t> * nan_sub_i; // this is an array full of row indices for extracted stuff
extern SA<size_t> * nan_sub_j; // this is an array full of col indices for extracted stuff
void nan_sub(size_t k); // apply nan under subscene window
void zero_sub(size_t k); //  apply zero under subscene window

// scene subsampling, parallelized by band
extern size_t mlk_scene_nb;  // infile nbands
extern size_t mlk_scene_nr; // infile nrows
extern size_t mlk_scene_nc; // infile ncols
extern size_t mlk_scene_nr2; // infile nrows
extern size_t mlk_scene_nc2; // infile ncols
extern size_t mlk_scene_np2;
extern float mlk_scene_scalef; // scaling factor
extern float * mlk_scene_dat; // float data output
extern string * mlk_scene_fn; // input filename
extern vector<int> * mlk_scene_groundref; // groundref indices

void multilook_scene(size_t k); // scene subsampling, parallelized by band

vector<vector<str> > read_csv(str fn, vector<str> & hdr); // csv reader

bool vin(vector<str> x, str a);
size_t vix(vector<str> x, str a);

size_t write_csv(str fn, vector<str> hdr, vector<vector<str> > lines);

std::string exec(const char* cmd);  

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
#endif
