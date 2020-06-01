/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere.. */
#include "newzpr.h"

zprInstance * myZpr = NULL;
zprInstance * myZpr2 = NULL;

int main(int argc, char ** argv){

  myZprManager = NULL;
  if(!myZprManager){
    myZprManager = zprManager::Instance(argc, argv);
  }

  myZpr = getNewZprInstance();

  /* Selection mode draw function */
  myZpr->zprPickFunc(my_pick);
  myZpr2 = getNewZprInstance();
  myZpr2->setRightOf(myZpr);

  glPlottable * a = new glBasicSphere(myZpr, 0, 0, 0, 0, 0, 1, 0.25, 10, 10);
  glPlottable * b = new glBasicSphere(myZpr2, 0, 0, 0, 0, 1, 1, 0.25, 10, 10);

  glArrow * a1 = new glArrow(myZpr, 0, 1, 0);
  a1->setXYZ(0,1,1,0,-1,0);

  glArrow * a2 = new glArrow(myZpr, 1, 1, 0);
  a2->setXYZ(0, 2, 3, 2, 2, 3);

  glArrow * a3 = new glArrow(myZpr, 1, 1, 1);
  a3->setXYZ(2, -2, 1, 1, -4, 0);

  /* Enter GLUT event loop */
  glutMainLoop();

  return 0;
}