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
  g++ -O3 sbo.cpp -o sbo

Running: 
  ./sbo file_from_SNAP.bin output_file.bin 4 

20220515: sbo_inplace.cpp [infile] [n-bytes per record]

this version modifies the file in-place! */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3)  err("sboi_inplace.exe [infile] [n-bytes per record]\n");
  
  FILE * g;
  stack<char> d;  // first-in, first-out structure to reverse byte order
  size_t n_b, i;
  int n = atoi(argv[2]);
  char * dat = read(argv[1], n_b); // read file into byte buffer
  g = wopen(argv[1]); // write back to same file
  char c;

  for0(i, n_b){
    d.push(dat[i]);
    if(d.size() >= n){
      while(d.size() > 0){
        c = d.top();
        d.pop();
        fputc(c, g);
      }
    }
  }
  fclose(g);
  return 0;
}
