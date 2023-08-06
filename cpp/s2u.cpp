/* updated 20230806 */
#include"misc.h"
int main(int argc, char ** argv){

  system("sentinel2_extract_swir.py");
  /*system("safe_unzip.py");
  system("sentinel2_stack_all.py 1");
*/
  system("clean");
  return 0;
}

