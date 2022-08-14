#include"misc.h"
int main(int argc, char ** argv){
  system("unzp");
  system("sentinel2_stack_all.py");
  system("clean");
  return 0;
}

