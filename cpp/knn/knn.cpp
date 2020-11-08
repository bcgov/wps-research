// data is the model! "Training" a model is a simple as assigning training data. // non-commutative distance-matrix algorithm

// if p is patch size, first write decimator to cut image modulo p, then take every n-th patch for train data (write train stack to width floor of sqrt number of train patches)
// write non-train stack to image of width floor of sqrt number of non-train patches

#include"../misc.h"
int main(int argc, char ** argv){
  printf("knn [input envi-type4 floating point stack bsq with gr] [# of one-hot encoded gr-ref maps] [input without gr to project onto] [patch size] [KMax] [K]\n");

  /* 1. code patches in linear segs, crash if alloc fail */

  /* 2. deduplicate? */

  /* 3. for each class, dmat calc (persist for params) -- incl. reflection in dcalc? (this is a non-square dmat) */

  /* 4. for each class, estimate density */

  /* 5. for each class, project density back into original image coordinates */

  /* 5. a enforce mathematical conditions for a probability density */



  return 0;
}
