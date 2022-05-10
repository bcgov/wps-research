#include"half.h"
#include<stdio.h>
#include<stdlib.h>

void err(const char * msg){
  printf("Error: %s\n", msg);
  exit(1);
}

void * alloc(size_t n){
  void * d = malloc(n);
  memset(d, '\0', n);
  return (void *)d;
}

int main(int argc, char ** argv){
  if(sizeof( uint32_t) != sizeof(float)) err("unexpected size");

  if(argc < 3){
    err("half2float.exe [input file] [output file]");
  }

  FILE * f = fopen(argv[1], "rb");
  if(!f) err("failed to open input file");
  fseek(f, 0, SEEK_END);
  size_t sz = ftell(f);
  printf("File size: %zu\n", sz);
  fclose(f);

  size_t nf = sz / 2; //sizeof(float);
  
  uint16_t * in = ( uint16_t *) alloc(sz);
  f = fopen(argv[1], "rb");
  fread(in, sizeof(uint16_t), nf, f);
  fclose(f);

  uint32_t * d = (uint32_t *)alloc(sizeof(uint16_t) * nf);
  
  size_t i;
  for(i = 0 ; i < nf; i++){
    d[i] = half_to_float(in[i]);
  }


  printf("+w %s\n", argv[2]);  
  f = fopen(argv[2], "wb");
  fwrite(d, sizeof(uint32_t), nf, f);
  fclose(f);

  return 0;
}



