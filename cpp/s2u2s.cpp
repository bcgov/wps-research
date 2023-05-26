/* 20230516  Extract all sentinel2 .SAFE folders to SWIR (re)sampled at 20m.

Note don't repeat steps if output files exist (check each step)
*/
#include"misc.h"
int main(int argc, char ** argv){
  int x = system("safe_unzip_gid.py");
  x = system("sentinel2_stack_all_20m.py 1 1");
  x = system("rm *swir*");
  x = system("clean");
  return 0;
}

