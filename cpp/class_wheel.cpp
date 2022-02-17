/* visualize a class map by going around the colour wheel */
#include"misc.h"
#include<vector>
#include<random>
int main(int argc, char ** argv){

  if(argc < 2) err("class_wheel [input binary file name]");
  str fn(argv[1]); // input file name
  cout << "input file name:" << fn << endl;
  str hfn(hdr_fn(fn)); // auto-detect header file name
  cout << "header file name:" << hfn << endl;
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  float r, g, b, h, s, v;
  if(nband != 1) err("this program defines results for 1-band images");
  
  float * dat = bread(fn, nrow, ncol, nband); // read floats
  map<float, size_t> count; // accumulate
  for0(i, np){
    if(count.count(dat[i]) < 1){
      count[dat[i]] = 0.;
    }
    count[dat[i]] += 1.;
  }

  priority_queue<float> pq;
  map<float, size_t>::iterator it;
  map<float, float> code_r, code_g, code_b;
  for(it = count.begin(); it != count.end(); it++){
    pq.push(it->first);
  }

  // number of codes: count.size()
  long int ci = 0;
  while(pq.size() > 0){
    float d = pq.top();
    pq.pop();

    if(d == 0.){
      code_r[d] = code_g[d] = code_b[d] = 0.;
    }
    else{
      s = v = 1.;
      v = ci / (float)(count.size() - 1);
      h = 60 + 300. * (float)ci / (float)(count.size() - 1);
      hsv_to_rgb(&r, &g, &b, h, s, v);

      cout << "d=" << d << " ci=" << ci << " h,s,v=" << h <<"," << s << "," << v<< endl;
      code_r[d] = r;
      code_g[d] = g;
      code_b[d] = b;
      ci ++; // next colour index
    }
  }

  /* add shuffling 20220216 */
  vector<size_t> shuf;
  for(i = 1; i < count.size(); i++) shuf.push_back(i);
  unsigned seed = 1;
  shuffle(shuf.begin(), shuf.end(), std::default_random_engine(seed));

  map<float, float> c_r, c_g, c_b;
  for(map<float, float>::iterator it = code_r.begin(); it != code_r.end(); it++){
    float d = it->first;
    if(d == 0.){
      c_r[d] = c_g[d] = c_b[d] = 0.;
    }
    else{
      float sv = shuf[(int)d - 1];
      c_r[d] = code_r[sv];
      c_g[d] = code_g[sv];
      c_b[d] = code_b[sv];
    }
  }

  str ofn(fn + str("_wheel.bin"));
  str ohfn(fn + str("_wheel.hdr"));
  hwrite(ohfn, nrow, ncol, 3); // rgb file: 3 bands

  // write colour encoded output
  FILE * outf = fopen(ofn.c_str(), "wb");
  for0(i, np){
    r = c_r[dat[i]];
    fwrite(&r, sizeof(float), 1, outf);
  }
  for0(i, np){
    g = c_g[dat[i]];
    fwrite(&g, sizeof(float), 1, outf);
  }
  for0(i, np){
    b = c_b[dat[i]];
    fwrite(&b, sizeof(float), 1, outf);
  }
  fclose(outf);
  return 0;
}
