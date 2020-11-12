// data is the model! "Training" a model is a simple as assigning training data. // non-commutative distance-matrix algorithm

// if p is patch size, first write decimator to cut image modulo p, then take every n-th patch for train data (write train stack to width floor of sqrt number of train patches)
// write non-train stack to image of width floor of sqrt number of non-train patches

#include"../misc.h"
int main(int argc, char ** argv){
  if(argc < 5){
    //note: the inputs are both patched. Input to project onto is patched, but without GR!
    printf("knn [input envi-type4 floating point stack bsq with gr (1)] [input image (gr not req'd): to project onto (2)] [KMax] [K]\n");
    exit(1);
  }
  // 1) is called "reference" (groundref + image combined)
  // 2) is called "source"
  // 3) output (transformed source) called product

  size_t i, j, k, nr, nc, nb, ps, fpp, bpp, ref_ps, ref_nb, K, K_max;
  str ref_f(argv[1]);
  str src_f(argv[2]);

  K = atoi(argv[4]);
  K_max = atoi(argv[3]); // how many things we persist

  ref_ps = int_read(ref_f + str("_ps"));
  ref_nb = int_read(ref_f + str("_nb"));
  ps = int_read(src_f + str("_ps"));
  nb = int_read(src_f + str("_nb"));
  if(ref_ps != ps) err("please ensure ref and src patch sizes are equal");
  if(ref_nb != nb) err("please ensure ref and src images have same number of image data bands, excl. ground-ref");

  printf("number of bands: %zu\n", ref_nb);
  printf(" patch length: %zu\n", ref_ps);
  fpp = ps * ps * nb; // floats per patch
  printf("floats per patch: %zu\n", fpp);
  bpp = fpp * sizeof(float);

  size_t ref_pfs = fsize(ref_f + str("_patch"));
  size_t src_pfs = fsize(src_f + str("_patch"));
  size_t ref_np = ref_pfs / bpp;
  size_t src_np = src_pfs / bpp;
  if(ref_pfs % bpp > 0 || src_pfs % bpp > 0){
    err("_patch file not multiple of bytes-per-patch");
  }

  printf("ref patches: %zu\n", ref_np);
  printf("src patches: %zu\n", src_np); // now list ref patch indices

  float * patch_label = float_read(ref_f + str("_patch_label"));

  #define mfs map<float, size_t>
  mfs c; // accumulate patch labels
  for0(i, ref_np){
    float L = patch_label[i];
    if(c.count(L) < 1) c[L] = 0;
    c[L] += 1;
  }
  cout << "patch labels:" << endl << c << endl;

  #define mfvs map<float, vector<size_t>>
  mfvs patches_per_label;
  for(mfs::iterator it = c.begin(); it != c.end(); it++) patches_per_label[it->first] = vector<size_t>();
 
  for0(i, ref_np) patches_per_label[patch_label[i]].push_back(i);




  // now, make sure to not stomp the output name!

  /* 3. for each class, dmat calc (persist for params) -- incl. reflection in dcalc? (this is a non-square dmat) */

  // list reference patch indices, for each label observed..
  // for each reference patch list, for each source patch, calculate a sorted truncated distance matrix "row"

  /* 4. for each class, estimate density (with respect to "reference") for each "source" data pixels.. */

  /* 5. for each class, project density (and max-density label) back onto "souce" image coordinates */

  /* 5. a enforce mathematical conditions for a probability density */

  /* calculate entropy maps (don't worry about accuracy) */

  return 0;
}

// might need to build image decimator latera
// obviously need to hierarchicalize this (fractalize)
