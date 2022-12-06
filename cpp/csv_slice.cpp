//---------------------------------------------------------------------------//
// slice.cpp: slice out select variables from CSV
// date: 20190124
//---------------------------------------------------------------------------//
#include<map>
#include"misc.h"
#include<ctime>
using namespace std;

int main(int argc, char ** argv){

  if(argc < 3) err("usage:\n\tslice.exe [infile] [Field name] .. [Field name n]");
  string filename(argv[1]);
  cout << "input file: " << filename << endl;

  ifstream infile(filename);
  if(!infile.is_open()) err(string("failed to open file: ") + filename);

  long unsigned int infile_pos;
  infile.seekg (0, infile.end);
  long unsigned int infile_len = infile.tellg();
  infile.seekg (0, infile.beg);

  string outfilename(filename + string("_slice.csv"));
  ofstream outfile(outfilename);
  if(!outfile.is_open()) err(string("failed to open file: ") + outfilename);
  cout << "output file: " << outfilename << endl;

  std::string line;
  vector<string> fields; // fields assumed to be listed 1st line of csv
  set<string> select;
  vector<int> selected;
  string delim(",");

  if(argc >= 3){
    for(int i = 2; i < argc; i++){
      string s_i(argv[i]);
      trim(s_i);
      // std::transform(s_i.begin(), s_i.end(), s_i.begin(), ::tolower); // to lower case
      select.insert(s_i);
    }
  }
  cout << "selected fields: " << select << endl;

  long int ci = 0; // line count
  long int n_f = 0;
  vector<int>::iterator it;

  time_t t0, t1; time(&t0);

  /* process the file line by line */
  while(std::getline(infile, line)){

    /* split csv line into separate fields */
    vector<string> w(split(line, ','));

    if(ci == 0){
      fields = w;
      n_f = w.size();

      for(int i = 0; i < n_f; i++){
        trim(w[i]);
        std::transform(w[i].begin(), w[i].end(), w[i].begin(), ::tolower); // to lower case

        if(select.count(w[i]) > 0){
          selected.push_back(i);
        }
      }
      if(w[w.size() - 1] == string("linefeed")){
        n_f -= 1;
      }
      for(it = selected.begin(); it != selected.end(); it++){
        if(it!=selected.begin()) outfile << delim;
        outfile << w[*it];
      }
    }
    else{
      outfile << "\n";

      if(w.size() != n_f){
        w = split_special(line);
        if(w.size() != n_f){
          cout << "w.size()" << w.size() << endl;
          cout << "n_f " << n_f << endl;
          cout << w << endl;
          // cout << "WARNING: " << (string("unexpected number of fields")) << end;
        }
      }
      for(it = selected.begin(); it != selected.end(); it++){
        if(it!= selected.begin()){
          outfile << delim;
        }
        outfile << w[*it];
      }
    }

    ci += 1;
    if(ci % 100000 == 0){
      infile_pos = infile.tellg();
      time(&t1);
      time_t dt = t1-t0;
      float mbps = (float)infile_pos / ((float)dt * (float)1000000.);
      float eta = (float)dt * ((float)infile_len - (float)infile_pos) / ((float)infile_pos);
      float p = 100. * (float)infile_pos / (float) infile_len;
      cout << "slic %" << p << " eta: " << eta << "s MB/s " << mbps << endl;
    }
  }

  outfile.close();
  infile.close();
  return 0;
}
