/* convert a class map to one-hot encoding. Revised 20220227 */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("class_onehot [input binary (classification) file name] [optional argument: minimum number of pixels for class to write]");

  size_t min_npix = 0;
  if(argc > 2) min_npix = (size_t)atoi(argv[2]);

  size_t nrow, ncol, nband, np, i, j;
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");
  size_t p1 = np / 100;
  printf("bread..\n");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);

  printf("accumulating..\n");
  map<float, size_t> count;
  for0(i, np){
    count[dat[i]] = count.count(dat[i]) < 1 ? 0 : count[dat[i]] + 1;
    if(i % p1 == 0){
      printf("Counting %zu / 100\n", i/p1);
    }
  }
  cout << count << endl;

  float d;
  float ci = 1.;
  map<float, float> lookup;
  map<float, size_t>::iterator it; // integral recoding
  for(it = count.begin(); it != count.end(); it++) lookup[it->first] = ci ++;
  cout << "lookup:" << lookup << endl;


  /* this part should be parallelised */
  size_t c_i = 1;
  for(it = count.begin(); it != count.end(); it++){
    float di = it->first;
    if(min_npix > 0) if(count[di] < min_npix) continue;
    printf("Write %e (size %zu) [%zu of %zu\n", di, count[di], c_i++, count.size()); 
    ci = lookup[di];
    str num(to_string(ci));
    num.erase(num.find_last_not_of('0') + 1, std::string::npos); // trailing 0
    num.erase(num.find_last_not_of('.') + 1, std::string::npos); // trailing .s
    str fnp(fn + str("_") + num);
    str ofn(fnp + ".bin");
    str ohfn(fnp + ".hdr");
  
    hwrite(ohfn, nrow, ncol, 1); // write header
    FILE * f = fopen(ofn.c_str(), "wb");
    for0(i, np){
      d = (dat[i] == di);
      fwrite(&d, sizeof(float), 1, f);
    }
    fclose(f);
  }
  free(dat);
  return 0;
}
