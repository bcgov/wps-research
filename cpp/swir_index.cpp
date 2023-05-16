/* 20230515 adapted from swir_increasing2.cpp */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("swir_index [input file name] # optional arg sat fraction");
  }

  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat, * b1, * b2, * b3;
  long int bi[3];

  str fn(argv[1]); /* binary files */
  str ofn(fn + "_swir_index.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));

  vector<string> s, t; /* band names + ones to use */
  t.push_back(str("945nm"));
  t.push_back(str("1610nm"));
  t.push_back(str("2190nm"));

  hread(hfn, nrow, ncol, nband, s);
  for0(i, 3) bi[i] = -1;
	np = nrow * ncol;
  n = s.size();

  str date_s;
  for0(i, n){
    for0(j, 3){
      if(contains(s[i], t[j])){
        bi[j] = i * np; /* found a required band */
        printf("bi[%zu]=%zu \"%s\"\n", j, bi[j], s[i].c_str());
        vector<string> w(split(s[i], ' '));
        date_s = w[0]; /* assume datetime string at beginning */
      }
    }
  }
  for0(i, 3) if(bi[i] < 0){
    printf("Missing band: %s\n", t[i].c_str());
    err("Missing band");
  }

  /* would be smarter to random read req'd bands only */
  dat = bread(fn, nrow, ncol, nband); /* read the data */
  b1 = &dat[bi[0]]; /* select the bands */
  b2 = &dat[bi[1]];
  b3 = &dat[bi[2]];
  out = falloc(np);
	float ** b = (float **) (void *)malloc(sizeof(float *) * 3);
	memset(b, '\0', sizeof(float *) * 3);
	b[0] = b1;
  b[1] = b2;
 	b[2] = b3;

	int nb = 0;
  FILE * f = fopen(ofn.c_str(), "wb");
	vector<str> bn;
	bn.clear();
  for0(j, 3){
    for0(i, j){
			printf("i %zu j %zu\n", i, j);		
			cout << "t[i]" << t[i] << " t[j]" << t[j] << endl;
			nb += 1;
			float * A = b[j];
		 	float * B = b[i];
      for0(k, np) out[k] = (A[k] - B[k]) / (A[k] + B[k]);
   		
			size_t n_write = fwrite(out, sizeof(float), np, f);
			printf("wrote %zu\n", n_write);
			cout << "t[i]" << t[i] << " t[j]" << t[j] << endl;
			str bns(str("("));
			bns += str(t[j]);
			bns += str(" - ");
			bns += str(t[i]);
			bns += str(") / (");
			bns += str(t[j]);
			bns += str(" + ");
			bns += str(t[i]);
			bns += str(")");
    	cout << "bns: " << bns << endl;
			bn.push_back(bns);
		}
  }
	fclose(f);
  hwrite(hf2, nrow, ncol, nb, 4, bn);
  free(dat);
  free(out);
  return 0;
}
