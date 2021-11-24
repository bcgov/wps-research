/* Dey, Subhadip; Bhattacharya, Avik; Frery, Alejandro C.; López-Martínez, Carlos (2020):
     A Model-free Four Component Scattering Power Decomposition for Polarimetric SAR Data.

To compile and run, e.g.:
   gcc MF3CC.c -o MF3CC.exe -lm
  ./MF3CC.exe C2

C impl. 20210422 by Ash Richardson, Senior Data Scientist, BC Wildfire Service */
#include<math.h>
#include<stdio.h>
#include<float.h>
#include<stdlib.h>
#include<memory.h>

#define for0(i,n) for(i = 0; i < n; i++) /* shorthand */

#define N_IN 4 /* number of input data files */
#define C11 0 /* C2 matrix input data indexing */
#define C12_re 1
#define C12_im 2
#define C22 3

const char* T_fn[] = {"C11.bin", /* T3 matrix input filenames */
                      "C12_real.bin",
                      "C12_imag.bin",
                      "C22.bin"};

float ** T; /* buffers for T3 matrix elements */
float ** T_f; /* buffers for filtered T3 matrix elements */
float ** out_d; /* output data buffers */

#define N_OUT 4 /* number of output files */
#define _theta_f 0 /* output data indexing */
#define _pd_f 1
#define _ps_f 2
#define _pv_f 3

const char * out_fn[] = {"Theta_CP.bin", /* output filenames */
                         "Pd_CP.bin",
                         "Ps_CP.bin",
                         "Pv_CP.bin"};

char sep(){
  #ifdef _WIN32
    return '\\'; /* windows path separator */
  #else
    return '/'; /* mac/linux/unix path sep */
  #endif
}

void err(const char * m){
  printf("Error: %s\n", m); /* print message */
  exit(1); /* quit */
}

#define MAX_ARRAYS 100 /* free after malloc: track allocations and auto-free after */
int n_arrays = 0;
void ** arrays;

void * alloc(size_t n){
  void * d = malloc(n); /* create array */
  if(!d) err("failed to allocate memory");
  memset(d, '\0', n); /* must touch memory on windows */
  arrays[n_arrays ++] = d; /* save pointer to free later */
  return d;
}

float * falloc(size_t n){
  return (float *)alloc(n * sizeof(float)); /* float array */
}

#define READ 1
#define WRITE 0

FILE * open(const char * fn, int mode){
  printf("+%s %s\n", mode?"r":"w", fn); /* open a file for read or write */
  FILE * f = fopen(fn, mode?"rb":"wb");
  if(!f){
    printf("Error: failed to open %s file: %s\n", mode?"input":"output", fn);
    exit(1);
  }
  return f;
}

void read_config(char * file_name, int * nrow, int * ncol){
  size_t x;
  char tmp[4096]; /* based on PolSARPro by Eric POTTIER and Laurent FERRO-FAMIL */
  FILE * f = open(file_name, READ);
  x = fscanf(f, "%s\n", tmp);
  x = fscanf(f, "%s\n", tmp); // number of rows
  *nrow = atoi(tmp);
  x = fscanf(f, "%s\n", tmp);
  x = fscanf(f, "%s\n", tmp);
  x = fscanf(f, "%s\n", tmp); // number of cols
  *ncol = atoi(tmp);
  fclose(f);
  printf("nrow %d ncol %d\n", *nrow, *ncol);
}

float * read(const char * file_name, size_t n_float){
  FILE * f = open(file_name, READ);
  float * d = falloc(n_float);
  size_t nr = fread(d, sizeof(float), n_float, f);
  if(nr != n_float){
    printf("Expected number of floats: %zu\n", n_float);
    printf("Number of floats read: %zu\n", nr);
    err("unexpected number of floats read");
  }
  fclose(f);
  return d; /* return array of floats we read in */
}

#define STR_MAX 4096 /* string variable length */
void hwrite(char * bfn, size_t nrow, size_t ncol, size_t nband){
  size_t i;
  char hfn[STR_MAX];
  size_t L = strlen(bfn);

  strcpy(hfn, bfn);
  hfn[L - 3] = 'h'; /* change ext from bin to hdr */
  hfn[L - 2] = 'd';
  hfn[L - 1] = 'r';

  FILE * f = open(hfn, WRITE);
  fprintf(f, "ENVI\n");
  fprintf(f, "samples = %zu\n", ncol);
  fprintf(f, "lines = %zu\n", nrow);
  fprintf(f,"bands = %zu\n", nband);
  fprintf(f, "header offset = 0\n");
  fprintf(f, "file type = ENVI Standard\n");
  fprintf(f, "data type = 4\n");
  fprintf(f, "interleave = bsq\n");
  fprintf(f, "byte order = 0\n");
  fprintf(f, "band names = {band 1");
  for0(i, nband - 1) fprintf(f, ",\nband %zu", i + 2);
  fprintf(f, "}\n");
  fclose(f);
}

float nan_to_num(float x){
  return isinf(x) ? 0 : (isnan(x)? FLT_MAX : x); /* replace NAN w zero, and infinity w large N */
}

int main(int argc, char ** argv){

  if(argc < 2)
    err("MF3CC.exe [input C2 directory] [optional chi_in param. (default -45 deg)]");

  float chi_in = -45.;
  char * path = argv[1]; /* T3 matrix data path */
  if(argc > 2) chi_in = atof(argv[2]); /* optional chi_in */
  int i, j, k, nrow, ncol, np, di, dj, ii, jj, x, ix, jx, nw;

  char fn[STR_MAX];
  strcpy(fn, path);
  fn[strlen(path)] = sep();
  strcpy(fn + strlen(path) + 1, "config.txt");
  read_config(fn, &nrow, &ncol); /* read image dimensions */
  np = nrow * ncol; /* number of px */

  arrays = (void *) malloc(sizeof(void *) * MAX_ARRAYS); /* array of pointers to free later */
  memset(arrays, '\0', sizeof(void *) * MAX_ARRAYS);
  n_arrays = 0;

  T = (float **) alloc(sizeof(float *) * N_IN); /* input file buffers */
  for0(k, N_IN){
    strcpy(fn, path);
    fn[strlen(path)] = sep();
    strcpy(fn + strlen(path) + 1, T_fn[k]); /* [path][sep][filename] e.g. T3/T11.bin */
    T[k] = read(fn, np); /* read each data band */
  }

  out_d = (float **) alloc(sizeof(float *) * N_OUT); /* output buffers */
  for0(i, N_OUT) out_d[i] = falloc(np); /* allocate output space */

  double c11, c12_r, c12_i, c22; /* intermediary variables */
  double trace2, s0, s1, s2, s3, SC, OC;
  double det, trace, trace3, m1, r, theta;
  double h, g, span, val, pd_f, ps_f, pv_f;
  float * c11_p, * c12_r_p, * c12_i_p, * c22_p;
  float * out_d_theta_f, * out_d_pd_f, * out_d_ps_f, * out_d_pv_f;

  c11_p = T[C11];
  c12_r_p = T[C12_re];
  c12_i_p = T[C12_im];
  c22_p = T[C22];

  out_d_theta_f = out_d[_theta_f];
  out_d_pd_f = out_d[_pd_f];
  out_d_ps_f = out_d[_ps_f];
  out_d_pv_f = out_d[_pv_f];

  for0(i, np){
    c11 = (double)c11_p[i];
    c12_r = (double)c12_r_p[i];
    c12_i = (double)c12_i_p[i];
    c22 = (double)c22_p[i];
    det = c11 * c22 - c12_i * c12_i - c12_r * c12_r;

    trace = c11 + c22;
    trace2 = trace * trace;
    m1 = sqrt(1. - (4. * det / trace2));

    s0 = c11 + c22;
    s1 = c11 - c22;
    s2 = 2. * c12_r;
    s3 = 2. * c12_i;
    if(chi_in >=0) s3 *= -1.;

    span = c11 + c22;
    SC = (s0 - s3) / 2.;
    OC = (s0 + s3) / 2.;
    theta = atan(((m1 * s0 * s3)) / ((SC * OC + (m1 * m1) * (s0 * s0))));

    pv_f  = (span * (1.0 - m1));
    ps_f = (((m1 * span * (1. + sin(2. * theta)) / 2.)));
    pd_f = (((m1 * span * (1. - sin(2. * theta)) / 2.)));
    theta *= (180. / M_PI);

    out_d_theta_f[i] = (float)theta;
    out_d_pd_f[i] = (float)pd_f;
    out_d_ps_f[i] = (float)ps_f;
    out_d_pv_f[i] = (float)pv_f;
  }

  FILE * out_f[N_OUT];
  for0(i, N_OUT){
    strcpy(fn, path);
    fn[strlen(path)] = sep();
    strcpy(fn + strlen(path) + 1, out_fn[i]);
    out_f[i] = open(fn, WRITE);
    hwrite(fn, nrow, ncol, 1); /* write envi header */
    nw = fwrite(out_d[i], sizeof(float), np, out_f[i]);
    if(nw != np) err("failed to write expected number of floats");
    fclose(out_f[i]);
  }

  for0(k, n_arrays) free(arrays[n_arrays]); /* free anything malloc'ed */
  free(arrays);
  return 0;
}
