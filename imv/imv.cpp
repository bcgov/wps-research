/* based on m3ta3: reimagination of a (late 2011, early 2012) primordial visualization library
that inspired further developments at UVic, CFS and elsewhere.. */ // todo: display original (untransformed) values (apply reverse map)
#include"util.h"
#include<fstream>
#include"newzpr.h"
#include<iostream>

// example function for parfor
void fx(size_t n){
  cprint(std::to_string(n));
}

int main(int argc, char ** argv){
  init_mtx(); // parfor(10, 15, fx); // parfor(20, 25, fx);
  groundref.clear();

  char * infile;
  if(sizeof(float) != 4){
    printf("Error: sizeof(float) != 4\n");
    exit(1);
  }
  if(argc < 2){
    printf("imv.cpp: [infile]\n");
    exit(1);
  }

  infile = argv[1];
  IMG_FN = string(infile);
  string mfn(string(argv[1]) + string(".ml"));
  cout << "multilook file name: " << mfn << endl;

  // analysis window size
  if(argc > 2){
    NWIN = atoi(argv[2]);
  }
  else{
    NWIN = 125;
  }
  WIN_I = WIN_J = 0;

  // window manager
  zprManager * myManager = zprManager::Instance(argc, argv);

  // get screen scale
  size_t width = glutGet(GLUT_SCREEN_WIDTH);
  size_t height = glutGet(GLUT_SCREEN_HEIGHT);
  size_t min = width > height ? height: width;
  min = 3 * min / 5; // can adjust scale here
  printf("min %d\n", min);

  size_t nr, nc, nb;
  string hfn(getHeaderFileName(string(infile)));
  
  // read band names
  groundref = parse_groundref_names(hfn);
  if(true){

  }

  exit(1);

  // get image scale
  parseHeaderFile(hfn, nr, nc, nb);
  printf(" infile: %s nrow %ld ncol %ld nband %ld\n", infile, nr, nc, nb);
  printf(" getFileSize %ld expected %ld\n", getFileSize(infile), nr * nc * nb * sizeof(float));
  size_t np = nr * nc;

  // determine scaling factor
  size_t imin = nr > nc ? nc: nr;
  printf("imin %d\n", imin);
  float scalef = (float)min / (float)imin;
  printf("scalef %f\n", scalef);
  SUB_SCALE_F = scalef;
  SUB_START_I = 0;
  SUB_START_J = 0;
  IMG_NR = nr;
  IMG_NC = nc;
  IMG_NB = nb;

  size_t nr2, nc2;
  nr2 = (int)ceil((float)nr * scalef);
  nc2 = (int)ceil((float)nc * scalef);
  printf("nr2 %d nc2 %d\n", nr2, nc2);

  // instead of loading the whole image, first load primary pane
  // SA<float> dat0(nr * nc * nb); // whole frame goes here
  SA<float> dat(nr2 * nc2 * nb); // this is the overview / multilook buffer
  size_t np2 = nr2 * nc2;
  SA<float> bb(np); // whole image, one-band buffer

  if(exists(mfn)){
    FILE * f = fopen(mfn.c_str(), "rb");
    if(!f){
      printf("Error: failed to open multilook file\n");
      exit(1);
    }
    fread(&dat[0], 1, np2 * nb * sizeof(float), f);
    fclose(f);
  }
  else{
    printf("scaling %d x %d image to %d x %d\n", nr, nc, nr2, nc2);
    FILE * f = fopen(infile, "rb");
    size_t nread = 0;
    for(size_t bi = 0; bi < nb; bi++){
      printf("fread band %zu/%d\n", bi + 1, nb);
      nread += fread(&bb[0], 1, sizeof(float) * np, f);

      for(size_t i = 0; i < nr; i++){
        int ip = (int)floor(scalef * (float)i);
        for(size_t j = 0; j < nc; j++){
          size_t jp = (int)floor(scalef * (float)j);
          size_t k = bi;
          dat[k * np2 + ip * nc2 + jp] = bb[(i * nc) + j];
          // last line could / should be += ?
        }
      }
    }
    if(nread != nr * nc * nb * sizeof(float)){
      printf("Error (imv.cpp): unexpected # of elements read (%d) expected (%d)\n", nread, nr * nc * nb);
      exit(1);
    }
    fclose(f);

    f = fopen(mfn.c_str(), "wb");
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
    load_sub_infile = string(infile);

    parfor(0, nb, load_sub);
  }
  IMG = NULL; // &dat0;
  SUB = &dat3;

  printf("min %d mm %d\n", min, mm);
  myImg b;
  b.initFrom(&dat3, mm, mm, nb);
  SUB_MYIMG = &b;

  // set up window for overview image
  zprInstance * myZpr = myManager->newZprInstance(nr2, nc2, nb);
  glImage * myImage = new glImage(myZpr, &a);
  myZpr->setTitle(string("Scene"));

  // set up window for fullres subset image
  zprInstance * myZpr2 = myManager->newZprInstance(mm, mm, nb);
  glImage * myImage2 = new glImage(myZpr2, &b);
  SUB_GLIMG = (void *)myImage2;
  myZpr2->setRightOf(myZpr);
  myZpr2->setTitle(string("Subset"));

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
	  float d= dat3.at(mm*mm*k + (WIN_I+i)*mm + (WIN_J + j));
	  dat4.at((NWIN * NWIN * k) + (NWIN *i) + j) = d;
        }
      }
    }
  }
  
  TGT_MYIMG = &c;

  //zprInstance *myZpr3 = myManager->newZprInstance(abs((long int)nr2 - (long int)mm), abs((long int)nr2 - (long int)mm), nb);
  zprInstance *myZpr3 = myManager->newZprInstance(NWIN, NWIN, nb);
  glImage * myImage3 = new glImage(myZpr3, &c);
  TGT_GLIMG = (void *)myImage3;
  myZpr3->setScreenPosition(0, nr2);
  myZpr3->setTitle(string("Analysis"));

  glutMainLoop();
  return 0;
}
