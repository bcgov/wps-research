/* calculate confusion matrix for a binary class map with respect to truth

Assume that no relabeling is required 

map tp, tn, fn, fp! 

*/
#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 3) err("binary_confusion [input binary class file name] [input ground reference file name]");

  // binary confusion matrix
  float cm [4] ={
	  0.,
	  0.,
	  0.,
	  0.};

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  size_t nrow2, ncol2, nband2;
  str cfn(argv[2]);
  str chfn(argv[2]);
  hread(chfn, nrow2, ncol2, nband2);

  // check the inputs are the same shape
  if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
    err("please check dimensional consistency of data in headers");
  }

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);
  float * cdat= bread(cfn, nrow, ncol, nband);

  // accumulate the data
  map<float, size_t> count;
  for0(i, np){
    if(count.count(dat[i]) < 1){
      count[dat[i]] = 0;
    }
    count[dat[i]] += 1;
  }

  float ci = 1.;
  map<float, float> lookup;
  for(map<float, size_t>::iterator it = count.begin(); it != count.end(); it++){
    lookup[it->first] = ci ++;
  }
/*
  hwrite(ohfn, nrow, ncol, 1); // write header

  float d;
  FILE * f = fopen(ofn.c_str(), "wb");

  for0(i, np){
	d = lookup[dat[i]];
	fwrite(&d, sizeof(float), 1, f);
  }
  fclose(f);
  */
  return 0;
}
