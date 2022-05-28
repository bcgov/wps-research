/* 202205276 convert iq format data (multidate stack) to PolSARPro scattering matrix (S2) format
..Separate folder for each date */
#include"misc.h"
void print(str s){
  cout << "[" << s << "]"<<endl;
}

int main(int argc, char ** argv){
  if(argc < 2){
    printf("Convert iq format data (time series stack) to PolSARPro s2 format\n");
    err("convert_iqstack_to_s2 [input file.bin]");
  }

  char * infile = argv[1];
  char * bands[4] = {"HH", "HV", "VH", "VV"};
  char * outb[4] = {"s11", "s12", "s21", "s22"};

  map<str, str> month;
  map<str, vector<str>> files;
  map<str, map<str, str>> pols;
  (month[str("Jan")] = str("01")), (month[str("Feb")] = str("02"));
  (month[str("Mar")] = str("03")), (month[str("Apr")] = str("04"));
  (month[str("May")] = str("05")), (month[str("Jun")] = str("06"));
  (month[str("Jul")] = str("07")), (month[str("Aug")] = str("08"));
  (month[str("Sep")] = str("09")), (month[str("Oct")] = str("10"));
  (month[str("Nov")] = str("11")), (month[str("Dec")] = str("12"));

  size_t i;
  str in_f(infile);
  str in_h(hdr_fn(in_f));
  size_t nrow, ncol, nband, np;
  hread(in_h, nrow, ncol, nband);
  vector<str> band_names(parse_band_names(in_h));
  np = nrow * ncol;

  vector<str>::iterator it;
  for(it = band_names.begin(); it != band_names.end(); it++){
    str x(*it);
    trim(x);
    vector<str> w(split(x, '_'));
    str mon(month[w[3].substr(2, 3)]);
    str year(w[3].substr(5, 4));
    str day(w[3].substr(0, 2));
    str iq(w[0]); // i or q
    str pp(w[1]); // HH, HV, VH, VV
    str out_dir(str(".") + psep() + year + mon + day + psep());

    if(files.count(out_dir) < 1){
      files[out_dir] = vector<str>();
      pols[out_dir] = map<str, str>();
    }
    str ifn(x + str(".bin"));
    files[out_dir].push_back(ifn);
    pols[out_dir][w[0] + w[1]] = ifn;
  }

  map<str, vector<str>>::iterator ti;
  for(ti = files.begin(); ti != files.end(); ti++){
    cout << "--------------------------------------------------\n";
    cout << ti->first << endl;
    cout << pols[ti->first] << endl;
  
    // open the output images
    FILE * of[4];
    for0(i, 4) of[i] = NULL;

    str odir(ti->first); //rgv[2]);
    if(!exists(odir))
      system((str("mkdir -p ") + odir).c_str());

    cout << "open s11 files:\n";
    for0(i, 4){
      str a(odir + str(outb[i]) + str(".bin"));
      print(a);
      
      of[i] = wopen(a.c_str());
      if(of[i] == NULL)
        err("failed to open output file");
    }

    str idir("./");
    for0(i, 4){
      str a(pols[odir][str("i") + str(bands[i])]);
      str b(pols[odir][str("q") + str(bands[i])]);
      print(a);
      print(b);

      FILE * f = fopen(a.c_str(), "rb");
      FILE * g = fopen(b.c_str(), "rb");
      if(f == NULL || g == NULL) err("failed to open input binary file");
  
      float * df = bread(a, nrow, ncol, nband); // read file
      float * dg = bread(b, nrow, ncol, nband); // read file
      float * dd = falloc(np * 2); // allocate floats

      size_t j, j2;
      for0(j, np){
        j2 = 2 * j;
        dd[j2] = df[j];
        dd[j2 + 1] = dg[j];
      }

      size_t nw = fwrite(dd, sizeof(float), np * 2, of[i]);
      if(nw != np * 2)
        err("unexpected write size");

      free(df);
      free(dg);
      free(dd);
      
      str c(odir + str(outb[i]) + str(".hdr"));
      hwrite(c, nrow, ncol, nband, 6);
    }
    for0(i, 4) fclose(of[i]);
  }
  return 0;
}
