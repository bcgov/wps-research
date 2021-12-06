/* 20211205 raster_relative_difference.cpp: relative difference operator for hyperspectral cubes
(adapted from raster_sub.cpp)

Output: 1) a third cube same dimension to the two inputs
	2) average % relative difference, by band (to stdout) */
#include"misc.h"

float f_max(float a, float b){
  return (a > b)? a: b;
}
float f_abs(float a){
  return (a > 0)? a: -a;
}

int main(int argc, char ** argv){
  printf("relative difference: input two hyperspectral cubes, dimension matching (output one similar cube)\n");
  if(argc < 4) err("raster_relative_difference.exe [raster cube 1] [raster cube 2 to subtract] [output cube]\n");

  str fn(argv[1]); // input image file name
  str fn2(argv[2]); // input image 2 file name
  if(!(exists(fn) && exists(fn2))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str hfn2(hdr_fn(fn2)); // input 2 header file name
  str ofn(argv[3]); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  hread(hfn, nrow, ncol, nband); // read header 1
  hread(hfn2, nrow2, ncol2, nband2); // read header 2
  if(nrow != nrow2 || ncol != ncol2 || nband != nband2)
    err("input image dimensions should match");

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat1 = bread(fn, nrow, ncol, nband); // assume they both fit in memory
  float * dat2 = bread(fn2, nrow, ncol, nband);

  float * avg = falloc(nband);
  float * N = falloc(nband);
  for0(i, nband) avg[i] = 0.;

  float d, e, f;
  for0(i, nrow){
    ix = i * ncol;
    for0(j, ncol){
      ij = ix + j;
      for0(k, nband){
	f = NAN;
        ik = ij + k * np;
	d = dat1[ik]; 
	e = dat2[ik];
	f = d * e;
	if(f != 0. && !isnan(f) && !isinf(f)){
          f = 2. * f_abs(d - e) / (f_abs(d) + f_abs(e));
	  avg[k] += f;
	  N[k] += 1.;
	}
        out[ik] = f;
      }
    }
  }

  printf("band index, average relative difference(%)\n");
  for0(k, nband){
    avg[k] /= N[k];
    printf("%d, %f = %e\n", k, 100. * avg[k], 100. * avg[k]);
  }

  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);

  /* output average % relative difference, by band*/ 
  
  free(dat1);
  free(dat2);
  free(out);
  free(avg);
  free(N);
  return 0;
}
