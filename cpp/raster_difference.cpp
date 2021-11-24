/* raster_difference.cpp: given a parameter N (bands per date)
 assuming within-date bands are in order, difference a stack band by band :

Supposing M bands of N bands per date, there are M/N dates..

..then this program produces M/N - 1 output bands (after - before) */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3){
    err("raster_difference.exe [multidate raster cube] [number of bands per date]\n");
  }

  size_t M = atoi(argv[2]);  // bands per date
  printf("bands per date: %d\n", M);

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name
  
  str ofn(str(argv[1]) + str("_difference.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name
  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat1 = bread(fn, nrow, ncol, nband);
 
  printf("number of dates: %d\n", nband/M);

  size_t N, K;
  for0(N, ((nband / M) - 1)){
     cout << N << endl; // subtract date N from date N+1
     for0(k, M){
	     for0(i, nrow){
	     for0(j, ncol){

			     // add on N * M to get this date's k...
			     // bands per date

			     // add on M to get the next date!
		     }
	     }
     }
  }
  return 0;
}

//  bwrite(out, ofn, nrow, ncol, nband);
//  hwrite(ohn, nrow, ncol, nband);
