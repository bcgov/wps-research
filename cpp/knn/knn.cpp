// data is the model! "Training" a model is a simple as assigning training data. // non-commutative distance-matrix algorithm

// if p is patch size, first write decimator to cut image modulo p, then take every n-th patch for train data (write train stack to width floor of sqrt number of train patches)
// write non-train stack to image of width floor of sqrt number of non-train patches

#include"../misc.h"
int main(int argc, char ** argv){
  
	//note: the inputs are both patched. Input to project onto is patched, but without GR! 
	printf("knn [input envi-type4 floating point stack bsq with gr (1)] [input image without gr to project onto (2)] [KMax] [K]\n");

 // 1) is called "reference" (groundref + image combined)
 // 2) is called "source"
 // 3) output (transformed source) called product

  size_t nr, nc, nb;
  str bfn(argv[1]);
  str bf2(argv[2]);

  size_t K = atoi(argv[4]);
  size_t kmax = atoi(argv[3]); // how many things we persist


  /* 3. for each class, dmat calc (persist for params) -- incl. reflection in dcalc? (this is a non-square dmat) */

  // list reference patch indices, for each label observed..
  // for each reference patch list, for each source patch, calculate a sorted truncated distance matrix "row"

  /* 4. for each class, estimate density (with respect to "reference") for each "source" data pixels.. */

  /* 5. for each class, project density (and max-density label) back onto "souce" image coordinates */

  /* 5. a enforce mathematical conditions for a probability density */

  /* calculate entropy maps */


  return 0;
}

// might need to build image decimator later
