/* based on m3ta3: reimagination of a (late 2011, early 2012) primordial visualization library
that inspired further developments at UVic, CFS and elsewhere.. 
todo: display original (untransformed) values (apply reverse map)?

spectra could have everything under window displayed! */ 
#include"util.h"
#include<fstream>
#include"newzpr.h"
#include<iostream>
#include <stdexcept>

extern vector<int> groundref;
extern size_t N_THREADS_IO;
extern string IMG_FN;
extern size_t NWIN;
extern size_t WIN_I;
extern size_t WIN_J;
extern vector<vector<str> > tgt_csv;
extern vector<str> tgt_csv_hdr;

extern vector<size_t> targets_i;
extern vector<size_t> targets_j;
extern vector<str> targets_label;

extern size_t SUB_START_I;
extern size_t SUB_START_J;
extern float SUB_SCALE_F;

extern size_t IMG_NR;
extern size_t IMG_NC;
extern size_t IMG_NB;

std::string strp(const std::string& str, const std::string& whitespace = " \t\r\n"){
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos) return ""; // no content
  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;
  return str.substr(strBegin, strRange);
}

#define message "imv.cpp [infile] # [analysis window size] [n bands groundref, 1-hot encoded, at end of bsq stack] [bands per frame] # [height max] [width max]"

int main(int argc, char ** argv){
  printf("Note: builtin classification method assumes input file: stack.bin\n");
  N_THREADS_IO = 4; // default number of threads for IO operations

  int n_groundref = 0;
  init_mtx();
  groundref.clear();
  IMG_FN = string("stack.bin"); // default image filename to load

  if(argc < 2) printf(message);
  else IMG_FN = string(argv[1]);
  if(argc > 3) n_groundref = atoi(argv[3]); // number of bands at end, to consider as groundref

  if(!exists(IMG_FN)){
    printf(message);
    err("failed to open input file"); // check if input file exists
  }
  string mfn(IMG_FN + string(".ml")); // look for saved multilook file
  cout << "multilook file name: " << mfn << endl;

  // analysis window size
  if(argc > 2) NWIN = atoi(argv[2]);
  else NWIN = 13; // 49; // obviously 49x49 for purposes of testing the extraction. Too big!

  if((NWIN - 1) % 2 != 0) err("analysis window size must be odd"); // assert analysis window size: odd
  WIN_I = WIN_J = 0; // window location? what is this?

  size_t xoff = 0;
  size_t yoff = 0;
  str xoff_f(IMG_FN + "_targets.csv_xoff");
  str yoff_f(IMG_FN + "_targets.csv_yoff");
  if(exists(xoff_f)){
    ifstream if1(xoff_f);
    str xoffs("");
    if1 >> xoffs;
    if1.close();
    xoff = atoi(xoffs.c_str());
  }
  if(exists(yoff_f)){
    ifstream if1(yoff_f);
    str yoffs("");
    if1 >> yoffs;
    if1.close();
    yoff = atoi(yoffs.c_str());
  }
  cout << "xoff " << xoff << " yoff " << yoff << endl;

  str targets_fn(IMG_FN + "_targets.csv");// load vector targets
  if(!exists(targets_fn)){
    FILE * f = fopen(targets_fn.c_str(), "wb"); // load point/window vector targets
    fprintf(f, "%s", "feature_id,row,lin,xoff,yoff");
    fclose(f);
  }

  tgt_csv = read_csv(targets_fn, tgt_csv_hdr); // read target point/window vector database
  if(!vin(tgt_csv_hdr, str("lin"))) err("req'd col missing: lin");
  if(!vin(tgt_csv_hdr, str("row"))) err("req'd col missing: lin");
  if(!vin(tgt_csv_hdr, str("feature_id"))) err("req'd col missing: lin");
  size_t row_i = vix(tgt_csv_hdr, str("row"));
  size_t lin_i = vix(tgt_csv_hdr, str("lin"));
  size_t fid_i = vix(tgt_csv_hdr, str("feature_id"));
  for(size_t i = 0; i < tgt_csv.size(); i++){
    size_t ti, tj; str tl;
    tj = atoi((tgt_csv[i])[row_i].c_str()) - xoff;
    ti = atoi((tgt_csv[i])[lin_i].c_str()) - yoff;
    tl = (tgt_csv[i])[fid_i];
    targets_i.push_back(ti);
    targets_j.push_back(tj);
    targets_label.push_back(tl);
  }

  zprManager * myManager = zprManager::Instance(argc, argv); // window manager class
  size_t width = glutGet(GLUT_SCREEN_WIDTH); // this section: get screen scale
  size_t height = glutGet(GLUT_SCREEN_HEIGHT);
  printf("glut width,height=(%zu, %zu)\n", width, height);


  if(argc > 5){
    size_t width_max = (size_t)atol(argv[5]);
    if(width_max < width) width = width_max;
  }

  if(argc > 6){
    size_t height_max = (size_t)atol(argv[6]);
    if(height_max < height) height = height_max;
  }

  if(!exists(".screen_min_x") ||
     !exists(".screen_min_y")){
    // screen dimensions not crossing boundaries between multiple displays
    int a = system("screen_dimension.py");
  }
  size_t width_min = (size_t)atol(readLines(str(".screen_min_x"))[0].c_str());
  size_t height_min = (size_t)atol(readLines(str(".screen_min_y"))[0].c_str());
  if(width_min < width) width = width_min;
  if(height_min < height) height = height_min;

  
  size_t min_wh = width > height ? height: width;
  printf("min_wh %f\n", (float)min_wh);
  size_t min = 3 * min_wh / 6; // can adjust scale here. has been 3/6, 3/5, 2/5
  printf("min %f\n", (float)min);
  size_t nr, nc, nb, nr2, nc2, np2;

  string hfn(getHeaderFileName(IMG_FN)); // this section: get image scale
  str my_user(exec("whoami")); // find out how many bands per date (for up arrow functionality)
  my_user = strp(my_user);
  str cmd(str("python3 /home/") + my_user + str("/GitHub/wps-research/py/envi_header_dates.py ") + hfn);
  cout << "[" << cmd << "]" << endl;
  str dates(exec(cmd.c_str()));
  vector<str> date_strings(split(dates, '\n'));
  int number_of_dates = date_strings.size(); // number of dates: hence number of bands per date
  if(number_of_dates < 1) number_of_dates = 1;

  parseHeaderFile(hfn, nr, nc, nb);
  printf("nrow %zu ncol %zu nband %zu\n", nr, nc, nb);
  if(nb == 1){
    str IMG_FN_NEW(IMG_FN + str("_x3.bin")); // deal with the case of one band input.. use as r, g and b!
    
    if(!exists(IMG_FN_NEW)){
      str cmd(str("cat ") + IMG_FN + str(" ") + IMG_FN + str(" ") + IMG_FN + str(" > ") + IMG_FN_NEW);
      cout << cmd << endl;
      system(cmd.c_str());
    }

    str IMG_HDR_NEW(IMG_FN + str("_x3.hdr"));
    if(!exists(IMG_HDR_NEW)){
      writeHeader(IMG_HDR_NEW.c_str(), nr, nc, 3);
    }
    IMG_FN = IMG_FN_NEW;
    hfn = IMG_HDR_NEW;
    parseHeaderFile(hfn, nr, nc, nb); // reload the data
  }

  size_t np = nr * nc;
  size_t min_wh_img = nr > nc ? nc: nr; // account for case that image is small!
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

  // scalef seems to need to be a few percent or so, less than one or the window transformations get thrown off..
  scalef = 0.95 * (float)(height > width ? width : height) / (float)(nr > nc? nr: nc); // some simplification of above code..
  if(nr < width && nr < height && nc < width && nc < height){
	  scalef = 1.; // account for "small image" case. 1-1 if can fit on screen!
  }

  if(nr < height && nc < width) scalef /= 2.;
 

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

 		bool use_multithread = N_THREADS_IO * np * 4 < 16000000000;

    parfor(0, nb, multilook_scene, use_multithread?N_THREADS_IO:1); // scene subsampling, parallelized by band
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
  else a.initFrom(&dat, nr2, nc2, nb);

  SUB_MYIMG = &a;

  // fullres display loading..
  printf("loading fullres data..\n");
  size_t mm = 3 * min / 2;
  if(nr < mm) mm = nr;
  if(nc < mm) mm = nc;
  mm = (7 * mm) / 7; // relative to full-scene overview window, make the full-res subscene window, a little smaller than it was before..

  SUB_MM = mm;
  size_t i_start = 0;
  size_t j_start = 0;
  SA<float> dat3(mm * mm * nb); // full-data subset
  printf("mm %d dat3.size %d\n", mm, dat3.size());
  SUB_I = new SA<size_t>(mm * mm);
  SUB_J = new SA<size_t>(mm * mm);
  for(size_t i = 0; i < mm * mm; i++) SUB_I->at(i) = SUB_J->at(i) = 0;
  if(true){
    load_sub_np = np; // parallelize on band idx (k)
    load_sub_nb = nb;
    load_sub_mm = mm;
    load_sub_i_start = i_start;
    load_sub_j_start = j_start;
    load_sub_nc = nc;
    load_sub_dat3 = &dat3[0];
    load_sub_infile = string(IMG_FN.c_str());
    load_sub_i = SUB_I;
    load_sub_j = SUB_J;
    parfor(0, nb, load_sub, N_THREADS_IO); // run parallel job
  }
  IMG = NULL; // &dat0;
  SUB = &dat3;

  printf("min %d mm %d\n", min, mm);
  myImg b;
  b.initFrom(&dat3, mm, mm, nb);
  SUB_MYIMG = &b;

  zprInstance * myZpr = myManager->newZprInstance(nr2, nc2, nb); // "Scene" / overview image WINDOW
  glImage * myImage = new glImage(myZpr, &a);
  SCENE_GLIMG = (void*)(glImage*)myImage;
  myZpr->setTitle(string("Scene "));

  zprInstance * myZpr2 = myManager->newZprInstance(mm, mm, nb); // fullres subset image WINDOW (subscene)
  glImage * myImage2 = new glImage(myZpr2, &b);
  SUB_GLIMG = (void *)myImage2;
  myZpr2->setRightOf(myZpr);
  myZpr2->setTitle(string("Subscene"));

  myImg c; // target window image setup
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
  zprInstance *myZpr3 = myManager->newZprInstance(max(222,nr2 - mm), max(222,nr2 - mm), nb); // analysis window
  glImage * myImage3 = new glImage(myZpr3, &c); // image object for analysis window data
  TGT_GLIMG = (void *)myImage3;
  myZpr3->setRightOf(myZpr2); //setScreenPosition(nc2, mm + 65); //nr2 + 65);
  myZpr3->setTitle(string("Analysis"));
  zprInstance * myZpr4 = myManager->newZprInstance(max(333,nr2 - mm), max(333,nr2 - mm), nb); // analysis window scatter window
  // glBasicSphere * s = new glBasicSphere(0, myZpr4, 0, 0, 0, 1, 1, 1, 1., 10, 10);

  vec3 v0(0,0,0); // origin. this section: unit square axes
  vec3 vx(1,0,0); vec3 vy(0,1,0); vec3 vz(0,0,1);
  glLine xL(myZpr4, v0, vx, 1, 0, 0);
  glLine yL(myZpr4, v0, vy, 0, 1, 0);
  glLine zL(myZpr4, v0, vz, 0, 0, 1);

  glPoints scatter(myZpr4, myImage3); // analysis window scatter window scatter plot
  myZpr4->setScreenPosition(nc2 + mm, nr2 + 65);
  myZpr4->setTitle(string("Scatter (analysis)"));

  zprInstance * myZpr5 = myManager->newZprInstance(max(333, nr2 - mm), max(333,nr2 - mm), nb); // preview scatter plot
  glLine xL2(myZpr5, v0, vx, 1, 0, 0); glLine yL2(myZpr5, v0, vy, 0, 1, 0);
  glLine zL2(myZpr5, v0, vz, 0, 0, 1);
  glPoints scatter2(myZpr5, myImage2); // scatter plot for: preview window
  myZpr5->setBelow(myZpr2); //setRightOf(myZpr4) ; // ScreenPosition(nc2, nr2 + 65);
  myZpr5->setTitle(string("Scatter(subsecne)")); // printf("glutMainLoop()\n");

  // spectra band window
  zprInstance * myZpr6 = myManager->newZprInstance(100, 250, 3);
  myZpr6->setBelow(myZpr); // setRightOf(myZpr5);
  myZpr6->setTitle(string("spectra"));
  glCurve spectraCurve(myZpr6, &spectra, 0, 1, 0);

  //histogram window..
  zprInstance * myZpr7 = myManager->newZprInstance(100, 250, 3);
  myZpr7->setBelow(myZpr6); //setScreenPosition(nc2 + mm, nr2 + mm + (nr2 - mm)); // 65 + nr2);
  myZpr7->setTitle(string("histogram"));
  glCurve histR(myZpr7, &hist_r, 1., 0., 0.);
  glCurve histG(myZpr7, &hist_g, 0., 1., 0.);
  glCurve histB(myZpr7, &hist_b, 0., 0., 1.);

  //myZpr6->setRightOf(myZpr7);
  myZpr4->setRightOf(myZpr5);
  myZpr6->setRightOf(myZpr4);
  myZpr7->setBelow(myZpr6);
  myZpr3->setRightOf(myZpr6);

  initLighting();

  bands_per_frame = nb / number_of_dates;
  printf("bands_per_frame %zu\n", bands_per_frame);
  if(argc > 4){
    bands_per_frame = atoi(argv[4]);
    printf("bands_per_frame %zu\n", bands_per_frame);
  }

  /* default vis section */
  if(nb == 66){
    myZpr2->setrgb(36, 48, 57); // for PRISMA (VNIR cube)
  }
  if(nb == 173){
    myZpr2->setrgb(43-1, 108-1, 171-1); // for PRISMA (SWIR cube)
  }
  if(nb == 239){
    //PRISMA combined sorted (SWIR) ?
  }
  if(nb >= 12){
    myZpr2->setrgb(12-1, 11-1, 10-1); // Sentinel-2 L2A sorted
  }
 
  myImage->rebuffer();
  myImage2->rebuffer();
  myImage3->rebuffer();

  glEnable(GL_DEPTH_TEST); // wow thanks to: https://learnopengl.com/Advanced-OpenGL/Depth-testing
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDepthMask(GL_FALSE);
  glDepthFunc(GL_LESS);

  glutMainLoop();
  printf("exit\n");
  return 0;
}
