#include"misc.h"
int main(int argc, char ** argv){
  system("unzp");
  system("sentinel2_stack_all.py 1");
  system("clean");
  return 0;
}

