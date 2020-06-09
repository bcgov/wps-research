#pragma once
#ifndef UTIL_H
#define UTIL_H

#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<memory.h>
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

/* shorthand for for loops from 0 to N */
#define for0(i,n) for(i = 0; i < n; i++)

using namespace std;
#define str string

void err(char * msg);

void err(string msg);

/*convert char to string: single character: interpret whitspace as space character */
string chartos(char s);

/*strip leading or trailing whitespace from a string*/
string strip(string s);

/*trim leading or trailing characters from a string*/
string trim(string s, char a);

long int getFileSize(std::string fn);
bool exists(string fn);

/* special case of split (for newline character) */
vector<string> split(string s, char delim, long int i);

/* split a string (a-la python) */
vector<string> split(string s, char delim);

vector<string> split(string s);
vector<string> readLines(string fn);
string getHeaderFileName( string fn);
vector<string> parseHeaderFile(string hfn, size_t & NRow, size_t & NCol, size_t & NBand);
void writeHeader(const char * filename, int NRows, int NCols, int NBand);

vector<string> parse_band_names(string fn); // read band names from header file
vector<int> parse_groundref_names(string fn); // indices of groundref bands

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
void parfor(size_t start_j, size_t end_j, void(*eval)(size_t));

// method for loading band of image, limited to subarea

extern size_t load_sub_np;
extern size_t load_sub_nb;
extern size_t load_sub_mm;
extern size_t load_sub_i_start;
extern size_t load_sub_j_start;
extern size_t load_sub_nc;
extern float * load_sub_dat3;
extern string load_sub_infile;

void load_sub(size_t k);

#endif
