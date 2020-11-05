/* based on m3ta3: reimagination of a (late 2011, early 2012) primordial visualization library
that inspired further developments at UVic, CFS and elsewhere.. */ // todo: display original (untransformed) values (apply reverse map)
#include"util.h"
#include<fstream>
#include"newzpr.h"
#include<iostream>

//#include <iostream>
#include <stdexcept>
//#include <stdio.h>
//#include <string>

std::string exec(const char* cmd){
  char buffer[128];
  std::string result = "";
  FILE* pipe = popen(cmd, "r");
  if (!pipe) throw std::runtime_error("popen() failed!");
  try{
    while(fgets(buffer, sizeof buffer, pipe) != NULL) result += buffer;
  }
  catch (...) {
    pclose(pipe);
    throw;
  }
  pclose(pipe);
  return result;
}

std::string strp(const std::string& str, const std::string& whitespace = " \t\r\n"){
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos) return ""; // no content
  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;
  return str.substr(strBegin, strRange);
}

int main(int argc, char ** argv){

  system("grep -n grep stretch newzpr.cpp"); // should be able to turn stretching on and off!

  int n_groundref = 0;
  init_mtx();
  groundref.clear();
  IMG_FN = string("stack.bin"); // default image filename to load

  if(argc < 2) printf("imv.cpp: [infile] optional: [nwin] [n bands groundref (one hot encoded, appearing at end of file..]\n");
  else IMG_FN = string(argv[1]);
  if(argc > 3) n_groundref = atoi(argv[3]); // number of bands at end, to consider as groundref

  if(!exists(IMG_FN)) err("failed to open input file"); // check if input file exists
  string mfn(IMG_FN + string(".ml")); // look for saved multilook file
  cout << "multilook file name: " << mfn << endl;

  // analysis window size
  if(argc > 2){
    NWIN = atoi(argv[2]);
  }
  else{
    NWIN = 49; // obviously this is for purposes of testing the extraction. Too big!
  }
  if((NWIN - 1) % 2 != 0) err("analysis window size must be odd"); // assert analysis window size: odd
  WIN_I = WIN_J = 0; // window location? what is this?

  zprManager * myManager = zprManager::Instance(argc, argv); // window manager class

  size_t width = glutGet(GLUT_SCREEN_WIDTH); // this section: get screen scale
  size_t height = glutGet(GLUT_SCREEN_HEIGHT);
  size_t min_wh = width > height ? height: width;
  printf("min_wh %f\n", (float)min_wh);
  size_t min = 3 * min_wh / 6; // can adjust scale here. has been 3/6, 3/5, 2/5
  printf("min %f\n", (float)min);
  size_t nr, nc, nb, nr2, nc2, np2;

  string hfn(getHeaderFileName(IMG_FN)); // this section: get image scale

  // find out how many bands per date (for up arrow functionality)
  str my_user(exec("whoami"));
  my_user = strp(my_user);
  str cmd(str("python3 /home/") + my_user + str("/GitHub/bcws-psu-research/py/envi_header_dates.py ") + hfn);
  cout << "[" << cmd << "]" << endl;
  str dates(exec(cmd.c_str()));
  vector<str> date_strings(split(dates, '\n'));
  int number_of_dates = date_strings.size();
 

  exit(1);

  // cout << "hfn: " << hfn << endl;
  parseHeaderFile(hfn, nr, nc, nb);
  // printf(" infile: %s nrow %ld ncol %ld nband %ld\n", IMG_FN.c_str(), nr, nc, nb);
  // printf(" getFileSize %ld expected %ld\n", getFileSize(IMG_FN.c_str()), nr * nc * nb * sizeof(float));
  size_t np = nr * nc;

  size_t min_wh_img = nr > nc ? nc: nr; // account for case that image is small!
  //printf("min_wh_img %f\n", (float)min_wh_img);
  if(min > min_wh_img) min = min_wh_img;
  printf("min %f\n", (float)min);

  vec_band_names = parse_band_names(hfn); // read band names
  groundref = parse_groundref_names(hfn, n_groundref);
  if(true){
    vector<int>::iterator it;
    for(it = groundref.begin(); it != groundref.end(); it++){
      cout << "groundref " << *it << endl;
    }
  }
  size_t imin = nr > nc ? nc: nr; // this section determine scaling factor
  float scalef = (float)min / (float)imin;
  printf("scalef %f\n", scalef);

  // width, height, nr, nc
  int height_lim = nr > height; // screen height too small for image
  int width_lim = nc > width; // screen width too small for image

  scalef = 0.95 * (float)(height > width ? width : height) / (float)(nr > nc? nr: nc); // some simplification of above code..
  if( nr < width && nr < height && nc < width && nc < height) scalef = 1.; // account for "small image" case. 1-1 if can fit on screen!

  SUB_START_I = SUB_START_J = 0;
  SUB_SCALE_F = scalef;
  IMG_NR = nr; IMG_NC = nc; IMG_NB = nb;
  nr2 = (int)ceil((float)nr * scalef);
  nc2 = (int)ceil((float)nc * scalef);
  printf("nr2 %d nc2 %d\n", nr2, nc2);

  // first load primary pane, instead of loading the whole image
  // SA<float> dat0(nr * nc * nb); // whole frame goes here
  np2 = nr2 * nc2;
  SA<float> dat(np2 * nb); // overview/ multilook image buffer
  SA<float> bb(np); // whole image, one-band buffer

  size_t mfn_fs = getFileSize(mfn);
  if(exists(mfn) && mfn_fs == np2 * nb * sizeof(float)){
    FILE * f = fopen(mfn.c_str(), "rb");
    if(!f) err("failed to open multilook file\n");
    fread(&dat[0], 1, np2 * nb * sizeof(float), f);
    fclose(f);
  }
  else{
    mlk_scene_nb = nb; // infile nbands. the section: scene subsampling, parallelized by band
    mlk_scene_nr = nr; // infile nrows
    mlk_scene_nc = nc; // infile ncols
    mlk_scene_nr2 = nr2; // infile nrows
    mlk_scene_nc2 = nc2; // infile ncols
    mlk_scene_np2 = np2; // number of pixels
    mlk_scene_scalef = scalef; // scaling factor
    mlk_scene_dat = &dat[0]; // float data output
    mlk_scene_fn = &IMG_FN; // input filename
    mlk_scene_groundref = &groundref; // groundref indices

    parfor(0, nb, multilook_scene); // scene subsampling, parallelized by band
    FILE * f = fopen(mfn.c_str(), "wb");
    fwrite(&dat[0], 1, np2 * nb * sizeof(float), f);
    fclose(f);
  }

  myImg a;
  if(nb == 1){
    SA<float> dat2(nr2 * nc2 * 3);
    size_t i, j, k;
    k = 0;
    for(i = 0; i < 3; i++){
      for(j = 0; j < nr2 * nc2; j++){
        dat2[k++] = dat[j];
      }
    }
    a.initFrom(&dat2, nr2, nc2, 3);
  }
  else{
    a.initFrom(&dat, nr2, nc2, nb);
  }

  // fullres display loading..
  printf("loading fullres data..\n");
  size_t mm = 3 * min / 2;
  if(nr < mm){
    mm = nr;
  }
  if(nc < mm){
    mm = nc;
  }
  SUB_MM = mm;
  size_t i_start = 0;
  size_t j_start = 0;
  SA<float> dat3(mm * mm * nb); // full-data subset
  printf("mm %d dat3.size %d\n", mm, dat3.size());
  if(true){
    // parameters for job. Parallelized on band idx
    load_sub_np = np;
    load_sub_nb = nb;
    load_sub_mm = mm;
    load_sub_i_start = i_start;
    load_sub_j_start = j_start;
    load_sub_nc = nc;
    load_sub_dat3 = &dat3[0];
    load_sub_infile = string(IMG_FN.c_str());

    // run parallel job
    parfor(0, nb, load_sub);
  }
  IMG = NULL; // &dat0;
  SUB = &dat3;

  printf("min %d mm %d\n", min, mm);
  myImg b;
  b.initFrom(&dat3, mm, mm, nb);
  SUB_MYIMG = &b;

  // set up window for overview image
  zprInstance * myZpr = myManager->newZprInstance(nr2, nc2, nb); // "Scene" / overview image
  glImage * myImage = new glImage(myZpr, &a);
  myZpr->setTitle(string("Scene "));

  // set up window for fullres subset image
  zprInstance * myZpr2 = myManager->newZprInstance(mm, mm, nb);
  glImage * myImage2 = new glImage(myZpr2, &b);
  SUB_GLIMG = (void *)myImage2;
  myZpr2->setRightOf(myZpr);
  myZpr2->setTitle(string("Subscene"));

  // target window setup

  myImg c;
  c.initBlank(NWIN, NWIN, nb);
  SA<float> dat4(NWIN * NWIN * nb); // target subset
  TGT = &dat4;

  if(true){
    size_t i, j, k;
    for0(k, IMG_NB){
      for0(i, NWIN){
        for0(j, NWIN){
          float d = dat3.at(mm * mm * k + (WIN_I + i) * mm + (WIN_J + j));
          dat4.at((NWIN * NWIN * k) + (NWIN * i) + j) = d;
        }
      }
    }
  }

  TGT_MYIMG = &c;
  zprInstance *myZpr3 = myManager->newZprInstance(NWIN, NWIN, nb); // analysis window
  glImage * myImage3 = new glImage(myZpr3, &c); // image object for analysis window data
  TGT_GLIMG = (void *)myImage3;
  myZpr3->setScreenPosition(0, nr2 + 65);
  myZpr3->setTitle(string("Analysis"));

  zprInstance * myZpr4 = myManager->newZprInstance(200, 200, nb); // analysis window scatter window
  // glBasicSphere * s = new glBasicSphere(0, myZpr4, 0, 0, 0, 1, 1, 1, 1., 10, 10);

  vec3 v0(0,0,0); // origin. this section: unit square axes
  vec3 vx(1,0,0); vec3 vy(0,1,0); vec3 vz(0,0,1);
  glLine xL(myZpr4, v0, vx, 1, 0, 0);
  glLine yL(myZpr4, v0, vy, 0, 1, 0);
  glLine zL(myZpr4, v0, vz, 0, 0, 1);

  glPoints scatter(myZpr4, myImage3); // analysis window scatter window scatter plot
  myZpr4->setRightOf(myZpr3); //myZpr4->setScreenPosition(nc2, nr2 + 65);
  myZpr4->setTitle(string("Analysis scatter"));

  zprInstance * myZpr5 = myManager->newZprInstance(200, 200, nb); // preview scatter plot
  glLine xL2(myZpr5, v0, vx, 1, 0, 0); glLine yL2(myZpr5, v0, vy, 0, 1, 0);
  glLine zL2(myZpr5, v0, vz, 0, 0, 1);

  glPoints scatter2(myZpr5, myImage2); // scatter plot for: preview window
  myZpr5->setRightOf(myZpr2) ; // ScreenPosition(nc2, nr2 + 65);
  myZpr5->setTitle(string("Subscene scatter")); // printf("glutMainLoop()\n");
  initLighting();

  bands_per_frame = nb / number_of_dates;
  
  glutMainLoop();
  return 0;
}
