#include"../misc.h" // tiler for ML app
size_t nrow, ncol, nband, np;

int main(int argc, char ** argv){
  if(argc < 4) err("tile [input envi-type4 floating point stack bsq with gr] [# of groundref classes at end] [patch size]\n");

  unsigned int ps, nref;
  str bfn(argv[1]);
  str hfn(hdr_fn(bfn));
  hread(hfn, nrow, ncol, nband);
  printf("nrow %zu ncol %zu nband %zu\n", nrow, ncol, nband);

  float * dat = bread(bfn, nrow, ncol, nband); // read input data
  np = nrow * ncol;
  ps = atoi(argv[3]); // patch size
  nref = atoi(argv[2]); // number of groundref classes one-hot encoded

  size_t i, j, m, n;
  if(ps >= nrow || ps >= ncol) err("patch size too big");
  if(nref >= nband) err("check number of groundref bands");

  unsigned int di, dj, k, ci;
  unsigned int nb = nband - nref;
  unsigned int floats_per_patch = ps * ps * nb;
  float * patch = falloc(sizeof(float) * floats_per_patch);

  FILE * f_patch = wopen("patch.dat");
  FILE * f_patch_i = wopen("patch_i.dat");
  FILE * f_patch_j = wopen("patch_j.dat");
  FILE * f_patch_label = wopen("patch_label.dat");

  map<float, unsigned int> count; // count labels on a patch
  size_t truthed = 0;
  size_t nontruthed = 0;

  for(i = 0; i < nrow - ps; i += ps){
    // start row for patch
    for(j = 0; j < ncol - ps; j += ps){
      // start col for patch
      ci = 0;
      for0(di, ps){
        m = i + di;
        for0(dj, ps){
          n = j + dj;
          for0(k, nb){
            patch[ci ++] = dat[(k * nrow * ncol) + (m * ncol) + n];
          }
        }
      }
      if(ci != ps * ps * nb) err("patch element count mismatch");
      // don't forget to calculate class for patch, by majority
      count.clear(); // mass for each label

      // count labels on a patch
      for0(di, ps){
        m = (i + di) * ncol;
        for0(dj, ps){
          n = (j + dj) + m;

	  int no_match = true;
          for0(k, nref){
	    unsigned int key = k + 1;

            float d = dat[((k + nb) * np) + n];
            if(d == 1.){
	      no_match = false;
              if(count.count(key) < 1) count[key] = 0;
              count[key] += 1;
            }
          }

	  if(no_match){
	    if(count.count(0) < 1) count[0] = 0;
	    count[0] += 1;
	  }
        }
      }
      float max_k = 0.;
      size_t max_c = 0;
      map<float, unsigned int>::iterator it; // lazy match
      for(it = count.begin(); it != count.end(); it++){

        if(it->first > 0 && it->second > max_c){
          max_c = it->second; // no leading class: doesn't count
          max_k = it->first;
        }
      }
      size_t n_match = 0;
      for(it = count.begin(); it != count.end(); it++) if(it->second == max_c) n_match ++;

      if(max_c > 0){
        truthed ++;
        cout << count << " max_c " << max_c << endl;
        if(n_match > 1) printf("\tWarning: patch had competing classes\n");
      }
      else{
        nontruthed ++;
      }

      fwrite(patch, sizeof(float), floats_per_patch, f_patch);
      fwrite(&max_k, sizeof(float), 1, f_patch_label);
    }
  }
  printf("\n");
  printf("nwin:          %zu\n", (size_t)ps);
  printf("image pixels:  %zu\n", np);
  printf("pix per patch: %zu\n",(size_t)(ps * ps));
  printf("approx patches:%zu\n", np / (ps * ps));
  printf("total patches: %zu\n", truthed + nontruthed);
  printf("truthed:       %zu\t\t[%.2f / 100]\n", truthed, 100. * (float)(truthed) / ((float)(truthed + nontruthed)));
  printf("nontruthed:    %zu\n", nontruthed);

  // patch label
  // patch i, patch j
  // patch start coord i, j (patch i, j * ps)
  // patch centre coord i, j
  // patch data

  fclose(f_patch);
  fclose(f_patch_i);
  fclose(f_patch_j);
  fclose(f_patch_label);
  free(patch);
  return 0;
}
