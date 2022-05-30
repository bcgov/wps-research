/* list points where a float mask is equal to 1. as input to qhull */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("binary_list [input binary class file name]");

  float d;
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);

  ofstream of;
  of.open("qhull.dat");

  str s(" ");
  size_t n = 0;
  for0(i, nrow){
    for0(j, ncol){
      if(dat[i * ncol + j] == 1.){
        n++;
      }
    }
  }
  of << to_string(2) << s << to_string(n) << s;
  for0(i, nrow){
    for0(j, ncol){
      if(dat[i * ncol + j] == 1.){
        of << i << s << j << s;
      }
    }
  }
  of << endl;
  of.close();

  ofstream pf;
  pf.open("alpha_shape_input_file.txt");
  pf << fn;
  pf.close();


  // run linux program qhull
  str r(exec("python3 ~/GitHub/wps-research/py/alpha_shape.py < qhull.dat"));
  strip(r);
  vector<str> lines(split(r, '\n'));
  //cout << "[" << r << "]" << endl;
  str X(lines[lines.size() -1]); // polygon
  trim(X, 'P'); trim(X, 'O'); trim(X, 'L'); trim(X, 'Y');
  trim(X, 'G'); trim(X, 'O'); trim(X, 'N');
  trim(X, ' '); trim(X, ' ');
  trim(X, '('); trim(X, '('); trim(X, ')'); trim(X, ')');
  cout << X<<endl;
  exit(1);

  /*
  int n_pts = atoi(lines[0].c_str()); // cout << lines.size() << endl;

  // confirm data consistency
  if(n_pts != lines.size() - 1) err("unexpected number of output lines");
  for0(i, n_pts){
    vector<str> x(split(lines[i+1], ' '));
    if(x.size() != 2) err("unexpected number of records");
    long int xi = atol(x[0].c_str());
    long int yi = atol(x[1].c_str());
  }

  // burn the points into dat and write out
  for0(i, np) dat[i] =0.;

  for0(i, n_pts){
    vector<str> x(split(lines[i+1], ' '));
    cout << s << s << x << endl;
    if(x.size() != 2) err("unexpected number of records");
    long int xi = atol(x[0].c_str());
    long int yi = atol(x[1].c_str());

    if(xi < 0 || xi > nrow || yi < 0 || yi > ncol){
      err("point out of bounds");
    }

    dat[xi * ncol + yi] = 1.;
  }
  */
  str ofn(fn + str("_hull.bin"));
  str ohn(fn + str("_hull.hdr"));
  hwrite(ohn, nrow, ncol, 1, 4); // write output header
  bwrite(dat, ofn, nrow, ncol, 1); // write binary data

  return 0;
}
