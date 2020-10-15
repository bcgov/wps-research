// data is the model! 

// if p is patch size, first write decimator to cut image modulo p, then take every n-th patch for train data (write train stack to width floor of sqrt number of train patches)
// write non-train stack to image of width floor of sqrt number of non-train patches

#include"misc.h"
int main(int argc, char ** argv){
  printf("knn [input envi-type4 floating point stack bsq with gr] [# of one-hot encoded gr-ref maps] [input without gr to project onto] [patch size] [KMax] [K]\n");
  
  /* 1. code patches in linear segs, crash if alloc fail */

  /* 2. deduplicate? */

  /* 3. dmat calc (persist for params) -- incl. reflection in dcalc? */

  /* 4. estimate density */



  


  
  return 0;
}
