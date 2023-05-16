/* 20230515 adapted from swir_index.cpp */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_index [input file name] # optional arg sat fraction");
  }

  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat;
  long int bi[3];

  str fn(argv[1]); /* binary files */
  str ofn(fn + "_index.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));

  vector<string> s;
  hread(hfn, nrow, ncol, nband, s);
	np = nrow * ncol;
  n = s.size();

  dat = bread(fn, nrow, ncol, nband); /* read the data */
  out = falloc(np);

	int nb = 0;
  FILE * f = fopen(ofn.c_str(), "wb");
	vector<str> bn;
	bn.clear();
  for0(i, nband){
    for0(j, i){
			nb += 1;
			float * A = &dat[np * j];
		 	float * B = &dat[np * i];
      for0(k, np) out[k] = (A[k] - B[k]) / (A[k] + B[k]);
			size_t n_write = fwrite(out, sizeof(float), np, f);
			str bns(str("("));
			bns += str(s[j]);
			bns += str(" - ");
			bns += str(s[i]);
			bns += str(") / (");
			bns += str(s[j]);
			bns += str(" + ");
			bns += str(s[i]);
			bns += str(")");
    	cout << "band name: " << bns << endl;
			bn.push_back(bns);
		}
  }
	fclose(f);
  hwrite(hf2, nrow, ncol, nb, 4, bn);
  free(dat);
  free(out);
  return 0;
}
