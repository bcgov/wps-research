// tile.cpp: tiling processing to feed into an ML-type code
#include"../misc.h"
size_t nrow, ncol, nband, np; // image attributes

void accumulate(map<float, unsigned int> &m, float key){
  if(m.count(key) < 1) m[key] = 0;
  m[key] += 1; // histogram on $\mathbb{R}^1
}

// could add "stride" parameter later (increase patches towards moving window)
int main(int argc, char ** argv){
  if(argc < 4) err("tile [input envi-type4 floating point stack bsq with gr] [# of groundref classes at end] [patch size]\n");

  str bfn(argv[1]);
  str hfn(hdr_fn(bfn));
  unsigned int ps, nref;
  hread(hfn, nrow, ncol, nband);
  printf("nrow %zu ncol %zu nband %zu\n", nrow, ncol, nband);

  float * dat = bread(bfn, nrow, ncol, nband); // read input data
  np = nrow * ncol; // number of pixels
  ps = atoi(argv[3]); // patch width / length: square patch
  nref = atoi(argv[2]); // number of groundref classes one-hot encoded

  size_t i, j, m, n;
  if(ps >= nrow || ps >= ncol) err("patch size too big");
  if(nref >= nband) err("check number of groundref bands");

  unsigned int di, dj, k, ci;
  unsigned int nb = nband - nref;
  unsigned int floats_per_patch = ps * ps * nb;
  float * patch = falloc(sizeof(float) * floats_per_patch);

  FILE * f_patch = wopen("patch.dat"); // patch data
  FILE * f_patch_i = wopen("patch_i.dat"); // start row for patch 
  FILE * f_patch_j = wopen("patch_j.dat"); // start col for patch
  FILE * f_patch_label = wopen("patch_label.dat"); // patch label

  size_t truthed = 0;
  size_t nontruthed = 0;
  map<float, unsigned int> count; // count labels on a patch

  for(i = 0; i < nrow - ps; i += ps){
    // start row for patch (stride parameter would be the step for this loop)
    for(j = 0; j < ncol - ps; j += ps){
      // start col for patch
      ci = 0;

      for0(di, ps){
        m = i + di;
        for0(dj, ps){
          n = j + dj;
          for0(k, nb) patch[ci ++] = dat[(k * nrow * ncol) + (m * ncol) + n];
        }
      }
      if(ci != ps * ps * nb) err("patch element count mismatch");
      count.clear(); // mass for each label

      for0(di, ps){
        m = (i + di) * ncol; // count labels on patch

        for0(dj, ps){
          n = (j + dj) + m;
          int no_match = true;

          for0(k, nref){
            if(dat[((k + nb) * np) + n] == 1.){
              no_match = false;
              accumulate(count, k + 1);
            }
          }
          if(no_match) accumulate(count, 0);
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
      else nontruthed ++;

      fwrite(&i, sizeof(size_t), 1, f_patch_i);
      fwrite(&j, sizeof(size_t), 1, f_patch_j);
      fwrite(&max_k, sizeof(float), 1, f_patch_label);
      fwrite(patch, sizeof(float), floats_per_patch, f_patch);
    }
  }
  printf("\n");
  printf(" nwin: %zu\n", (size_t)ps);
  printf(" image pixels: %zu\n", np);
  printf(" pix per patch: %zu\n",(size_t)(ps * ps));
  printf(" approx patches:%zu\n", np / (ps * ps));
  printf(" total patches: %zu\n", truthed + nontruthed);
  printf(" truthed: %zu\t\t[%.2f / 100]\n", truthed, 100. * (float)(truthed) / ((float)(truthed + nontruthed)));
  printf(" nontruthed: %zu\n", nontruthed);

  fclose(f_patch); // patch data
  fclose(f_patch_i); // patch start i
  fclose(f_patch_j); // patch start j
  fclose(f_patch_label); // patch label
  free(patch);
  return 0;
}
