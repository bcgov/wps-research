/* g++ forest.cpp -lgsl -w -o forest -O4
*/
#include<map>
#include<string>
#include<math.h>
#include<vector>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<iostream>
#include<memory.h>
#include"matrix2.h"
#include<algorithm>
#include<gsl/gsl_errno.h>
#include<gsl/gsl_roots.h>
#include"../psp/psp.h"
using namespace std;
/* Limit of Iterative Times */
#define MAXTIMES 50
#define cf complex<float>

#define strip trim

/* define indexes for each channel of information from T4 matrix */
#define r11 0
#define r12 1
#define i12 2
#define r13 3
#define i13 4
#define r14 5
#define i14 6
#define r22 7
#define r23 8
#define i23 9
#define r24 10
#define i24 11
#define r33 12
#define r34 13
#define i34 14
#define r44 15

/* shorthand for for loops from 0 to N */
#define for0(i,n) for(i = 0; i < n; i++)

void err(string msg){
  cout << "Error: " << msg << endl;
  exit(1);
}

FILE * wopen(string s){
  FILE * ret = fopen(s.c_str(), "w");
  if(!ret){
    printf("Error: failed to open file: %s\n", s.c_str());
    err("failed to open file");
  }
  return ret;
}

size_t size(FILE * f){
  fseek(f, 0L, SEEK_END);
  size_t sz = ftell(f);
  rewind(f);
  return sz;
}

size_t fsize(string fn){
  FILE * f = fopen(fn.c_str(), "rb");
  if(!f) return (size_t) 0;
  size_t fs = size(f);
  fclose(f);
  return fs;
}

bool exists(string fn){
  return fsize(fn) > 0;
}

string hdr_fn(string fn, bool create=false){
  string hfn(fn + string(".hdr"));
  if(exists(hfn)) return hfn;
  int s = fn.size();
  string b(fn);
  b = b.substr(0, s - 4);
  string hfn2(b + string(".hdr"));
  if(exists(hfn2)) return hfn2; // don't create two headers for the same file

  if(create == true) return hfn2;
  else err(string("could not find header at: ") + hfn + string(" or at: ") + hfn2);
  return(hfn);
}

string hdr_fn(string fn){
  return hdr_fn(fn, false);
}

int hwrite(string hfn, size_t nrow, size_t ncol, size_t nband){
  //cout << "hwrite" << endl;
  cout << "+w " << hfn << " nrow " << nrow << " ncol " << ncol << " nband " << nband << endl;
  ofstream hf(hfn);
  if(!hf.is_open()) err("failed to open header file for writing");
  hf << "ENVI" << endl;
  hf << "samples = " << ncol << endl;
  hf << "lines = " << nrow << endl;
  hf << "bands = " << nband << endl;
  hf << "header offset = 0" << endl;
  hf << "file type = ENVI Standard" << endl;
  hf << "data type = 4" << endl;
  hf << "interleave = bsq" << endl;
  hf << "byte order = 0" << endl;
  hf << "band names = {band 1";
  size_t i;
  for0(i, nband - 1) hf << ",\n" << "band " << i + 2;
  hf << "}" << endl;
  hf.close();
  return 0;
}

inline void * alloc(size_t nb){
  void * d = malloc(nb);
  if(!d){
    printf("%zu\n", nb);
    err("failed to allocate memory");
  }
  memset(d, '\0', nb);
  return d;
}

float * falloc(size_t nf){
  return (float *) alloc(nf * (size_t)sizeof(float));
}

const vector<string> var_n{
  string("P"), string("theta"), string("R"), string("M"), string("delta"), string("entropy"),
  string("alpha"), string("mD"), string("aD"), string("dA"), string("raleighp")
};
vector<string> ofn;
vector<string> ohn;
map<string, int> lookup; // map variable names to indices in the band-sequential array

size_t NRow, NCol;

void bwrite(float * d, string bfn, size_t nrow, size_t ncol, size_t nband){
  size_t nf = nrow * ncol * nband;
  FILE * f = wopen(bfn.c_str()); //fopen(bfn.c_str(), "wb");
  if(!f) err("bwrite(): failed to open output file");
  fwrite(d, sizeof(float), nf, f);
  fclose(f);
}

double Hd;
double func(double m, void *param){
  double rat = 1./((2.*m)+1.);
  double l3r = 1./( log(3.));
  double ret = ((-rat) *log(rat)*l3r ) - ((2.*m*rat)*log( m*rat)*l3r);
  ret = ret - Hd;
  return ret;
}

double dfunc(double m, void *param){
  /* f'(x):
  MATLAB CODE:
  >> syms m l3r;
  >> f= - (( 1/ (2*m +1)) * log( 1/ (2*m +1)) *l3r) - ( (2*m / (2*m +1)) * log( m/(2*m+1)) * l3r);
  >> simple(diff(f,m))

  2*l3r*(log(1/(2*m+1))-log(m/(2*m+1)))/(2*m+1)^2 */
  double l3r = 1./ ( log(3.));
  return 2.*l3r*(log(1./((2.*m)+1.))-log(m/((2.*m)+1)))/( ((2.*m)+1) * ((2.*m)+1));
}

/* simultaneous eval of f and df */
void fdfunc(double x, void *param, double *ret_f, double *ret_df){
  *ret_f = func(x, param);
  *ret_df = dfunc(x, param);
  return;
}

float ** out_buf;

const vector<string> t4fn = {
  string("T11.bin"),
  string("T12_real.bin"),
  string("T12_imag.bin"),
  string("T13_real.bin"),
  string("T13_imag.bin"),
  string("T14_real.bin"),
  string("T14_imag.bin"),
  string("T22.bin"),
  string("T23_real.bin"),
  string("T23_imag.bin"),
  string("T24_real.bin"),
  string("T24_imag.bin"),
  string("T33.bin"),
  string("T34_real.bin"),
  string("T34_imag.bin"),
  string("T44.bin")
};

FILE * ropen(string fn){
  printf("+r %s\n", fn.c_str());
  FILE * ret = fopen(fn.c_str(), "rb");
  if(!ret){
    printf("Error: failed to open file: %s\n", fn.c_str());
    err(string("failed to open file"));
  }
  return ret;
}

float * bread(string bfn, size_t nrow, size_t ncol, size_t nband){
  FILE * f = ropen(bfn);
  size_t nf = nrow * ncol * nband;
  float * dat = falloc(nf);
  size_t nr = fread(dat, nf * (size_t)sizeof(float), 1, f);
  if(nr != 1){
    printf("bread(%s)\n", bfn.c_str());
    err("misc.cpp: bread(): failed to read data");
  }
  fclose(f);
  return dat;
}

bool not_space(int data){
  return !std::isspace(data);
}

static inline void ltrim(std::string &s){
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),not_space));
}

// trim from end (in place)
static inline void rtrim(std::string &s){
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

static inline string ltrim(string & s, string chars){
  s.erase(0, s.find_first_not_of(chars));
  return s;
}

static inline string rtrim(string & s, string chars){
  std::size_t found = s.find_last_not_of(chars);
  if (found!=std::string::npos) s.erase(found + 1);
  return s;
}

static inline void trim(string & s, string chars){
  ltrim(s, chars);
  rtrim(s, chars);
}

// trim from both ends (in place)
static inline void trim(string &s){
  ltrim(s);
  rtrim(s);
}

/* split a string (a-la python) */
vector<string> split(string s, char delim){
  trim(s);
  bool extra = (s[s.size() - 1] == delim);
  std::vector<std::string> ret;
  std::istringstream iss(s);
  std::string token;
  while(getline(iss,token,delim)) ret.push_back(token);
  if(extra) ret.push_back(string(""));
  return ret;
}

/* split a string (a-la python) */
vector<string> split(string s){
  trim(s);
  const char delim = ',';
  bool extra = (s[s.size() -1] == delim);
  std::vector<std::string> ret;
  std::istringstream iss(s);
  string token;
  while(getline(iss,token,delim)) ret.push_back(token);
  vector<string>::iterator it;
  for(it = ret.begin(); it != ret.end(); it++) trim(*it);
  if(extra) ret.push_back(string(""));
  return ret;
}

// read header file
size_t hread(string hfn, size_t & nrow, size_t & ncol, size_t & nband){
  cout << "hread:" << hfn << endl;
  string line;
  size_t data_type;
  vector<string> words;
  ifstream hf(hfn);
  data_type = nrow = ncol = nband = 0;
  if(hf.is_open()){
  }
  else{
    err(string("failed to open header file: ") + hfn);
  }
  while(getline(hf, line)){
    words = split(line, '=');
    if(words.size() == 2){
      strip(words[0]);
      string w(words[0]);
      size_t n = (size_t) atoi(words[1].c_str());
      if(w == string("samples")) ncol = n;
      if(w == string("lines")) nrow = n;
      if(w == string("bands")) nband = n;
      if(w == string("data type")) data_type = n;
    }
  }
  cout << "hread: " << hfn << " nrow: " << nrow << " ncol: " << ncol << " nband: " << nband << " data_type: " << data_type << endl;
  hf.close();
  return data_type;
}

float ** t4;

cf T(size_t ix, int i){
  /* retrieve T4 matrix element at position ix, subscript i */
  switch(i){
    case 11: return cf(t4[r11][ix], 0.); break;
    case 22: return cf(t4[r22][ix], 0.); break;
    case 33: return cf(t4[r33][ix], 0.); break;
    case 44: return cf(t4[r44][ix], 0.); break;
    case 12: return cf(t4[r12][ix], t4[i12][ix]); break;
    case 13: return cf(t4[r13][ix], t4[i13][ix]); break;
    case 14: return cf(t4[r14][ix], t4[i14][ix]); break;
    case 21: return cf(t4[r12][ix], -t4[i12][ix]); break;
    case 23: return cf(t4[r23][ix], t4[i23][ix]); break;
    case 24: return cf(t4[r24][ix], t4[i24][ix]); break;
    case 31: return cf(t4[r13][ix], -t4[i13][ix]); break;
    case 32: return cf(t4[r23][ix], -t4[i23][ix]); break;
    case 34: return cf(t4[r34][ix], t4[i34][ix]); break;
    case 41: return cf(t4[r14][ix], -t4[i14][ix]); break;
    case 42: return cf(t4[r24][ix], -t4[i24][ix]); break;
    case 43: return cf(t4[r34][ix], -t4[r34][ix]); break;
    default: printf("Error: invalid subscript on T4 matrix\n"); exit(1); break;
  }
}

int main(int argc, char ** argv){
  if(argc < 3){
    printf("Forest.cpp: forest parameters implemented from Shane Cloude's lecture notes January 11, 2010 by Ash Richardson. Reimplemented 20170623, 20240219\n");
    printf("Usage: forest [indir] [outdir]\n");
    exit(1);
  }
  const double PI = atan(1.0)*4;

  size_t i, j, k, np, ix, nband;
  float P, theta, R, M, delta, entropy, alpha, mD, aD, dA;
  float * _P, * _theta, * _R, * _M, * _delta, * _entropy;
  float * _alpha, * _mD, * _aD, * _dA, * _raleighp;

  char * indir = argv[1];
  char * outdir = argv[2];

  vector<FILE*> t4f;
  for0(i, t4fn.size()){
    string f(indir + string("/") + t4fn[i]);
    t4f.push_back(ropen(t4fn[i]));
  }

  string t0fn(string(indir) + string("/") + t4fn[0]);
  t0fn = hdr_fn(t0fn, false);
  cout << t0fn << endl;
  hread(t0fn, NRow, NCol, nband);

  t4 = (float **)(void*)malloc(sizeof(float*) * t4fn.size());
  memset(t4, '\0', sizeof(float*) * t4fn.size());
  for0(i, t4fn.size()){
    string f(indir + string("/") + t4fn[i]);
    t4[i] = bread(f, NRow, NCol, 1);
  }

  for0(i, var_n.size()) lookup[var_n[i]] = i; // create the index-lookup for variable names

  //T4 T(INPUT, indir);
  //T.getDimensions(NRow, NCol);
  np = NRow * NCol;

  float *** t = matrix3d_float(4, 4, 2); //input matrix
  float *** v = matrix3d_float(4, 4, 2); //eigenvector matrix
  float * lambda = vector_float(4);
  float p[4], Alpha[4], H;
  float * m;

  FILE ** out_files = (FILE**)(void*)malloc(sizeof(FILE*) * var_n.size());
  memset(out_files, '\0', sizeof(FILE*) * var_n.size());

  out_buf = (float **)(void*)malloc(sizeof(float*) * var_n.size());
  memset(out_buf, '\0', sizeof(float*) * var_n.size());

  for0(i, var_n.size()){
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << i + 1;
    std::string s = string(outdir) + string("/") + string("K") + ss.str() + string("_") + var_n[i] + string(".bin");
    ofn.push_back(s);
    ohn.push_back(hdr_fn(s, true));
    //out_files[i] = wopen(s.c_str());
    hwrite(ohn[i], NRow, NCol, 1);
    out_buf[i] = falloc(np);
    for0(j, np) out_buf[i][j] = 0.;
  }

  _P = out_buf[lookup[string("P")]];
  _theta = out_buf[lookup[string("theta")]];
  _R = out_buf[lookup[string("R")]];
  _M = out_buf[lookup[string("M")]];
  _delta = out_buf[lookup[string("delta")]];
  _entropy = out_buf[lookup[string("entropy")]];
  _alpha = out_buf[lookup[string("alpha")]];
  _mD = out_buf[lookup[string("mD")]];
  _aD = out_buf[lookup[string("aD")]];
  _dA = out_buf[lookup[string("dA")]];
  _raleighp = out_buf[lookup[string("raleighp")]];

  printf("Was able to allocate memory...\n");


  printf("\n ");

  /* Define Solver */
  gsl_function f;
  gsl_root_fsolver *workspace_f = gsl_root_fsolver_alloc(gsl_root_fsolver_bisection);
  gsl_function_fdf fdf;
  gsl_root_fdfsolver *workspace;
  gsl_set_error_handler_off();

  int times, status;
  double x, x_l, x_r;
  float at13, at33, at23, rt23, mHHmVV2, mHHpVV2, mHV2;

  for(i = 0; i < NRow; i++){
    for(j = 0; j < NCol; j++){
      ix = i * NCol + j;
      if(ix % (np / 1000) == 0) printf("%d/100\n", (int)((float)ix/((float)np)*100.));

      at13 = abs(T(ix, 13));
      at33 = abs(T(ix, 33));
      at23 = abs(T(ix, 23));
      rt23 = real(T(ix, 23));
      //calculate K=1 (model as single layer) p.23 of Shane Cloude's presentation January 11, 2010.
      mHHpVV2 = 2*at13*at13/at33;
      mHHmVV2 = 2*at23*at23/at33;
      mHV2 = 0.5*at33;

      P = 0.5 * ( mHHpVV2 + mHHmVV2 + 4.*mHV2);
      theta = 0.25 * atan2( 4.* rt23 , (mHHmVV2 - 4. * mHV2) );
      R = ( mHHmVV2 - 4.* mHV2 ) / (mHHmVV2 + 4. * mHV2);
      M = (mHHmVV2 + 4.* mHV2) / mHHpVV2;
      delta = (float) T(ix, 12).angle(); //arg(T[12]);

      //write parameters
      _P[ix] = P;
      _theta[ix] = theta;
      _R[ix] = R;
      _M[ix] = M;
      _delta[ix] = delta;

      //m = &(T.pixel[0]); /* Average complex coherency matrix determination*/
      t[0][0][0] = eps + t4[r11][ix];
      t[0][0][1] = 0.;
      t[0][1][0] = eps + t4[r12][ix];
      t[0][1][1] = eps + t4[i12][ix];
      t[0][2][0] = eps + t4[r13][ix];
      t[0][2][1] = eps + t4[i13][ix];
      t[0][3][0] = eps + t4[r14][ix];
      t[0][3][1] = eps + t4[i14][ix];

      t[1][0][0] = eps + t4[r12][ix];
      t[1][0][1] = eps - t4[i12][ix];
      t[1][1][0] = eps + t4[r22][ix];
      t[1][1][1] = 0.;
      t[1][2][0] = eps + t4[r23][ix];
      t[1][2][1] = eps + t4[i23][ix];
      t[1][3][0] = eps + t4[r24][ix];
      t[1][3][1] = eps + t4[i24][ix];

      t[2][0][0] = eps + t4[r13][ix];
      t[2][0][1] = eps - t4[i13][ix];
      t[2][1][0] = eps + t4[r23][ix];
      t[2][1][1] = eps - t4[i23][ix];
      t[2][2][0] = eps + t4[r33][ix];
      t[2][2][1] = 0.;
      t[2][3][0] = eps + t4[r34][ix];
      t[2][3][1] = eps + t4[i34][ix];

      t[3][0][0] = eps + t4[r14][ix];
      t[3][0][1] = eps - t4[i14][ix];
      t[3][1][0] = eps + t4[r24][ix];
      t[3][1][1] = eps - t4[i24][ix];
      t[3][2][0] = eps + t4[r34][ix];
      t[3][2][1] = eps - t4[i34][ix];
      t[3][3][0] = eps + t4[r44][ix];
      t[3][3][1] = 0.;

      Diagonalisation(4, t, v, lambda);

      for (k = 0; k < 4; k++)
      if (lambda[k] < 0.) lambda[k] = 0.;

      for (k = 0; k < 4; k++) {
        // Unitary eigenvectors
        Alpha[k] = acos(sqrt(v[0][k][0] * v[0][k][0] + v[0][k][1] * v[0][k][1]));
        p[k] = lambda[k] / (eps + lambda[0] + lambda[1] + lambda[2] + lambda[3]);
        if (p[k] < 0.) p[k] = 0.;
        if (p[k] > 1.) p[k] = 1.;
      }

      alpha = 0;
      entropy = 0;
      for(k = 0; k < 3; k++){
        alpha += p[k]*Alpha[k];
        entropy += -p[k]*(log(p[k])/log(3.));
      }

      //ROOT SOLVER
      Hd = entropy;

      //printf("F solver: %s\n", gsl_root_fsolver_name(workspace_f));
      f.function = &func;
      f.params = 0;

      /* set initial interval */
      x_l = 0.0+eps;
      x_r = 1.0-eps;

      /* set solver */
      gsl_root_fsolver_set(workspace_f, &f, x_l, x_r);

      /* main loop */
      for0(times, MAXTIMES){
        //for(times = 0; times < MAXTIMES; times++)
        status = gsl_root_fsolver_iterate(workspace_f);

        x_l = gsl_root_fsolver_x_lower(workspace_f);
        x_r = gsl_root_fsolver_x_upper(workspace_f);
        //printf("%d times: [%10.3e, %10.3e]\n", times, x_l, x_r);

        status = gsl_root_test_interval(x_l, x_r, 1.0e-13, 1.0e-20);
        if(status != GSL_CONTINUE){
          //printf("Status: %s\n", gsl_strerror(status));
          //printf("\n Root = [%25.17e, %25.17e]\n\n", x_l, x_r);
          break;
        }
      }

      mD = x_l;
      aD = mD * PI / ( (2.* mD ) + 1);
      if(isnan(t4[r11][ix])){
        mD = NAN;
        aD = NAN;
      }

      dA = alpha - aD;
      _entropy[ix] = entropy;
      _alpha[ix] = alpha;
      _mD[ix] = mD;
      _aD[ix] = aD;
      _dA[ix] = dA;
    }
  }

  double sigma2=0.;
  double d;
  double count = 0.;

  for0(i, NRow){
    for0(j, NCol){
      ix = i * NCol + j;
      if(isnan(t4[r11][ix])){
      }
      else{
        d = _dA[ix];
        sigma2 += d*d;
        count += 1.;
      }
    }
  }
  sigma2 = sigma2 / ( 2. * ((float)count));
  printf("Sigma %e Sigma^2 %e\n", sqrt(sigma2), sigma2);
  printf("Mean %e\n", sqrt(sigma2)*sqrt(PI / 2.));
  printf("In degrees:\n");
  double conv = 180. / PI;
  float rp = 0;

  printf("Sigma %e Sigma^2 %e\n", conv*sqrt(sigma2), conv*sigma2);
  printf("Mean %e\n", conv*sqrt(sigma2)*sqrt(PI / 2.));
  printf("Computing raleigh probability..\n");
  bool warned = false;
  for0(i, NRow){
    for0(j, NCol){
      ix = i * NCol + j;
      d = _dA[ix];
      if(d < 0) warned = true;
      _raleighp[ix] = (isnan(t4[r11][ix]))? NAN: (d / sigma2) * exp ( - d * d / ( 2. * sigma2));
    }
  }

  if(warned) printf("WARNING: deltaalpha was less than 0.\n");
  for0(i, var_n.size()) bwrite(out_buf[i], ofn[i], NRow, NCol, 1);
  cout << "NRow " << NRow << " NCol " << NCol << endl;

  string xx("raster_stack.py ");
  for0(i, var_n.size()) xx += ofn[i] + string(" ");
  xx += string("stack.bin");
  cout << xx << endl;
  int retcode = system(xx.c_str());

  gsl_root_fsolver_free(workspace_f);
  printf("\rdone");
  return 0;
}
