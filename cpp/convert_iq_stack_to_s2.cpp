/* 202205276 convert iq format data (multidate stack) to PolSARPro scattering matrix (S2) format
..Separate folder for each date */
#include"misc.h"

void print(str s){
  cout << "[" << s << "]"<<endl;
}

int main(int argc, char ** argv){
  map<str, str> month;
  (month[str("Jan")] = str("01")), (month[str("Feb")] = str("02"));
  (month[str("Mar")] = str("03")), (month[str("Apr")] = str("04"));
  (month[str("May")] = str("05")), (month[str("Jun")] = str("06"));
  (month[str("Jul")] = str("07")), (month[str("Aug")] = str("08"));
  (month[str("Sep")] = str("09")), (month[str("Oct")] = str("10"));
  (month[str("Nov")] = str("11")), (month[str("Dec")] = str("12"));

  if(argc < 2){
    printf("Convert iq format data (time series stack) to PolSARPro scattering matrix (S2) format\n");
    err("convert_iqstack_to_s2 [input file.bin]");
  }
  char * infile = argv[1];
  str in_f(infile);
  str in_h(hdr_fn(in_f));

  vector<str> band_names(parse_band_names(in_h));
  vector<str>::iterator it;
  for(it = band_names.begin(); it != band_names.end(); it++){
    str x(*it);
    vector<str> w(split(x, '_'));
    cout << w << endl;
  }

  exit(1);
  /*
  char * o_d = argv[1];

  int i, ci;
  size_t nr, nc, nb;
  size_t nr2, nc2, nb2;

  char * bands[4] = {
    "HH",
    "HV",
    "VH",
    "VV"};

  char * outb[4] = {
    "s11",
    "s12",
    "s21",
    "s22"};

  ci = 0;  // check the input image sizes match
  str idir(argv[1]);
  rtrim(idir, str("/"));
  for0(i, 4){
    str a(idir + sep() + str("i_") + str(bands[i]));
    str b(idir + sep() + str("q_") + str(bands[i]));

    hread(a + str(".bin.hdr"), nr2, nc2, nb2);
    if(ci++ == 0){
      nr = nr2; nc = nc2; nb = nb2;
    }
    if(nr != nr2 || nc != nc2 || nb != nb2) err("image dimensions mismatch");
    hread(b + str(".bin.hdr"), nr2, nc2, nb2);
    if(nr != nr2 || nc != nc2 || nb != nb2) err("image dimensions mismatch");
  }

  if(nb != 1) err("only one-band images supported");
  size_t np = nr * nc; // number of pxls

  // open the output images
  FILE * of[4];
  for0(i, 4) of[i] = NULL;

  str odir(argv[2]);
  rtrim(odir, str("/"));
  for0(i, 4){
    str a(odir + sep() + str(outb[i]) + str(".bin"));
    print(a);
    of[i] = wopen(a.c_str());
    if(of[i] == NULL) err("failed to open output file");
  }

  for0(i, 4){
    str a(idir + sep() + str("i_") + str(bands[i]) + str(".bin")); // print(a);
    str b(idir + sep() + str("q_") + str(bands[i]) + str(".bin")); // print(b);

    FILE * f = fopen(a.c_str(), "rb");
    FILE * g = fopen(b.c_str(), "rb");
    if(f == NULL || g == NULL) err("failed to open input binary file");

    float * df = bread(a, nr, nc, nb); // read file
    float * dg = bread(b, nr, nc, nb); // read file
    float * dd = falloc(np * 2); // allocate floats

    size_t j, j2;
    for0(j, np){
      j2 = 2 * j;
      dd[j2] = df[j];
      dd[j2 + 1] = dg[j];
    }

    size_t nw = fwrite(dd, sizeof(float), np * 2, of[i]);
    if(nw != np * 2) err("unexpected write size");

    free(df);
    free(dg);
    free(dd);

    str c(odir + sep() + str(outb[i]) + str(".hdr"));
    hwrite(c, nr, nc, nb, 6);
  }

  for0(i, 4) fclose(of[i]);

  str cmd(str("cp -v ") + idir + sep() + str("*.txt ") + odir + sep());
  str cm2(str("cp -v ") + idir + sep() + str("*.xml ") + odir + sep());
  system(cmd.c_str());
  system(cm2.c_str());

  print(cmd);
  */
  return 0;
}
