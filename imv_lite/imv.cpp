/* simple one-to one image viewer */
#include"util.h"
#include<fstream>
#include"newzpr.h"
#include<iostream>
#include <stdexcept>

int main(int argc, char ** argv){
  IMG_FN = string("stack.bin"); // default image filename to load
  if(argc < 2) printf("imv.cpp: [infile]\n");
  else IMG_FN = string(argv[1]);
  if(!exists(IMG_FN)) err("failed to open input file"); // check if input file exists

  string hfn(getHeaderFileName(IMG_FN)); // this section: get image scale
  size_t nr, nc, nb, np;
  parseHeaderFile(hfn, nr, nc, nb);
  np = nr * nc;

  zprManager * myManager = zprManager::Instance(argc, argv); // window manager class
  SA<float> dat(np * nb); // the image data..
  SA<float> bb(np); // whole image, one-band buffer

  FILE * f = fopen(IMG_FN.c_str(), "rb");
  if(!f) err("failed to open input file\n");
  fread(&dat[0], sizeof(float), np * nb, f);

  myImg a;
  a.initFrom(&dat, nr, nc, nb);

  SCENE_MYIMG = &a;
  zprInstance * myZpr = myManager->newZprInstance(nr, nc, nb); // "Scene" / overview image WINDOW
  glImage * myImage = new glImage(myZpr, &a);
  SCENE_GLIMG = (void*)(glImage*)myImage;
  myZpr->setTitle(string("Scene "));

  myImage->rebuffer();

  glutMainLoop();
  return 0;
}
