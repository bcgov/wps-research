/* 20220419 swap byte order for pure-binary data file:
Contains only linear stream of binary records:
   # of bytes per record = n
(last version 20190510)

Number of bytes per record:
    (e.g. 4 for ENVI type-4 (32-bit IEEE standard floating point)
i.e. 32 bytes / (8 bits/byte) = 4 bytes

Usage:
  sboi.exe [infile] [outfile] [n-bytes per record]

Compilation:
	g++ -O3 sbo.cpp -o sbo.exe
*/
#include<stack>
#include<stdio.h>
#include<stdlib.h>
using namespace std;

void err(const char * s){
  printf("Error: %s\n", s);
  exit(1);
}

int main(int argc, char ** argv){
  if(argc < 4)
    err("sboi.exe [infile] [outfile] [n-bytes per record]\n");

  char c;
  FILE * f, *g;
  stack<char> d;
  int n = atoi(argv[3]);
  long unsigned int nb = 0;
  f = fopen(argv[1], "rb");
  g = fopen(argv[2], "wb");
  if(!f || !g)
    err("please check input and output files\n");

  while(fread(&c, 1, 1, f) == 1){
    nb ++;
    d.push(c);

    if(d.size() >= n){
      while(d.size() > 0){
        c = d.top(); // first-in, first-out structure reverses the byte order
        d.pop();
        fputc(c, g);
      }
    }
  }
  printf("bytes read %ld\n", nb);
  return 0;
}