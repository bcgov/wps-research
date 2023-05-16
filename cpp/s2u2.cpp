/* 20230515 */
#include"misc.h"
int main(int argc, char ** argv){
  system("unzp");
  system("sentinel2_stack_all_20m.py 1");
  system("clean");
  return 0;
}

