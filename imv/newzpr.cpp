/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere.. */
#include <math.h>
#include "newzpr.h"
#include <stdlib.h>
#include <memory.h>
#include <iostream>
using namespace std;

// subset window data
size_t SUB_START_I;
size_t SUB_START_J;
float SUB_SCALE_F;
size_t SUB_MM;
size_t IMG_NR;
size_t IMG_NC;
size_t IMG_NB;
SA<float> * IMG = NULL;
SA<float> * SUB = NULL;
myImg * SUB_MYIMG = NULL;
void * SUB_GLIMG = NULL;
string IMG_FN;

// target window data
size_t NWIN; // square window length/width
size_t WIN_I; // target selection index (window upper-left corner)
size_t WIN_J;
SA<float> * TGT = NULL;
myImg * TGT_MYIMG = NULL;
void * TGT_GLIMG = NULL;

// groundref detection
vector<int> groundref;
vector<string> vec_band_names;

extern zprManager * myZprManager = NULL;

void GLERROR(){
  cout << "GLERROR\n";
  /*
  GLenum code = glGetError();
  while(code != GL_NO_ERROR){
    printf("%s\n",(char *) gluErrorString(code));
    code = glGetError();
  }
  */
}

void glImage::drawMeUnHide(){
  hideMe = false;
  Update = true;
  drawMe();
  hideMe = true;
  Update = false;
}

zprManager * zprManager::Instance(int argc, char *argv[]){
  cout <<"Check for singleton instance..\n";
  if(!myZprManager){
    myZprManager = new zprManager(argc,argv);
  }
  return(myZprManager);
}

void zprManager::mark(){
  int i;
  for(i = 0; i < this->nextZprInstanceID; i++){
  }
}

void zprInstance::mark(){
  refreshflag = true;
  std::vector<glPlottable *>::iterator it;
  int i=0;
  for(it = myGraphics.begin(); it != myGraphics.end(); it++){
  }
}

void two_percent(float & min, float & max, SA<float> * b){
  // not actually 2%, gasp!
  priority_queue<float> q;
  float * d = b->elements;
  unsigned int n_two = floor(0.02 * ((float)b->size()));
  unsigned int i;
  for(i = 0; i < b->size(); i++){
    q.push(d[i]);
  }

  for(i = 0; i < n_two; i++){
    q.pop();
  }
  max = q.top();

  while(q.size() > n_two){
    q.pop();
  }
  min = q.top();

  while(q.size() > 0){
    q.pop();
  }
  printf("two_p n=%zu min %f max %f\n", b->size(), min, max);
}

void glImage::rebuffer(){
  myBi = parentZprInstance->myBi;
  int NRow = image->NRow; int NCol = image->NCol;
  printf("glImage::rebuffer() %d %d %d nr %d nc %d\n", myBi->at(0), myBi->at(1), myBi->at(2), NRow, NCol);
  SA< SA<float> * > * FB = image->getFloatBuffers();
  SA<float> * b1 = FB->at(myBi->at(0));
  SA<float> * b2 = FB->at(myBi->at(1));
  SA<float> * b3 = FB->at(myBi->at(2));
  float max1, max2, max3, min1, min2, min3, r1, r2, r3;
  r1 = r2 = r3 = 1.;
  min1 = min2 = min3 = 0.;

  two_percent(min1, max1, b1); // so the 2p stretch happens in the secondary buffer (this one)
  two_percent(min2, max2, b2);
  two_percent(min3, max3, b3);
  r1 = 1./(max1 - min1);
  r2 = 1./(max2 - min2);
  r3 = 1./(max3 - min3);

  printf("myZprInstanceID %d r1 %f r2 %f r3 %f min1 %f min2 %f min3 %f\n", parentZprInstance->myZprInstanceID, r1, r2, r3, min1, min2, min3);

  float r, g, b;
  long int i, j, k, ri, m;
  int class_i, n_class, gi;
  k = 0;
  m = 0;

  for(i = 0; i < NRow; i++){
    ri = NRow - i - 1;
    for(j = 0; j < NCol; j++){
      r = r1 * (b1->at(ri, j) - min1);
      r = (r<0. ? 0.: r);
      r = (r>1. ? 1.: r);
      dat->at(k++) = r;

      g = r2 * (b2->at(ri, j) - min2);
      g = (g<0. ? 0.: g);
      g = (g>1. ? 1.: g);
      dat->at(k++) = g;

      b = r3 * (b3->at(ri, j) - min3);
      b = (b<0. ? 0.: b);
      b = (b>1. ? 1.: b);
      dat->at(k++) = b;

      class_i = -1;
      n_class = 0;
      for(gi = 0; gi < groundref.size(); gi++){
	// for(gi = groundref.begin(); gi != groundref.end(); gi++)
        if(FB->at(groundref[gi])->at(ri, j) > 0.){
          class_i = gi + 1;
          n_class += 1;
        }
      }
      // set the class label to class_i if only one label. set to 0 if none. set to -1 if more than one!
      class_label->at(m++) = (n_class == 1. ? (float)class_i : (n_class > 1 ? -1. : 0.));
    }
  }
  Update = false;
  printf("\tdone rebuffer\n");
}

zprInstance * zprManager::newZprInstance(int NROW, int NCOL, int NBAND){
  int myWindowWidth = NCOL;
  int myWindowHeight = NROW;
  int myDims = NBAND;

  dprintf("zprManager::newZprInstance(%d,%d)", myWindowWidth, myWindowHeight);

  zprInstance * ret = NULL;
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(myWindowWidth, myWindowHeight);
  int newWindowID = glutCreateWindow("zprInstance");
  if(myGlutWindowIDs.count(newWindowID)){
    cout << "Error (zpr.h): glut window with this glut window ID is already initialized.\n";
    exit(1);
  }
  myGlutWindowIDs.insert(newWindowID);
  ret = new zprInstance(nextZprInstanceID, newWindowID, this, NROW, NCOL, myDims);
  ret->myKnnClusteringInstance = NULL;
  ret->myPickNames.clear();
  ret->zprInit();
  (*myZprInstances)[nextZprInstanceID] = ret;
  ret->myZprInstanceID = nextZprInstanceID;
  myZprWindowID_AsFunctionOfGlutID[ newWindowID] = nextZprInstanceID;
  myGlutID_AsFunctionOfZprWindowID[nextZprInstanceID] = newWindowID;
  ret->setScreenPosition(0,0);

  glutDisplayFunc(myZprDisplay);
  glutSpecialFunc(myZprSpecial);
  glutSpecialUpFunc(myZprSpecialUp);
  glutKeyboardFunc(myZprKeyboard);
  glutKeyboardUpFunc(myZprKeyboardUp);

  glutIdleFunc(myZprIdle);
  ret->pick = my_pick;
  glScalef(0.25,0.25,0.25);
  ret->zprSelectionFunc(myZprDrawGraphics);
  nextZprInstanceID++;

  dprintf("zprManager::newZprInstance: newWindowID (%d), myZprInstanceID (%d), nextZprInstanceID (%d), myZprWindowID_AsFunctionOfGlutID[ newWindowID](%d), myGlutID_AsFunctionOfZprWindowID[nextZprInstanceID](%d)", newWindowID, ret->myZprInstanceID, nextZprInstanceID, myZprWindowID_AsFunctionOfGlutID[newWindowID], myGlutID_AsFunctionOfZprWindowID[ret->myZprInstanceID]);
  return(ret);
}

void zprManager::zprReshape( int w, int h){
  (myZprInstances->at(this->getActiveZprInstanceID()))->zprReshape(w,h);
}

void zprManager::zprMouse(int button, int state, int x, int y){
  (myZprInstances->at(this->getActiveZprInstanceID()))->zprMouse(button, state, x, y);
}

void zprManager::zprMotion(int x, int y){
  (myZprInstances->at(this->getActiveZprInstanceID()))->zprMotion(x,y);
}

void zprManager::_pick(GLint name){
  (myZprInstances->at(this->getActiveZprInstanceID()))->_pickme(name);
}

void zprManager::zprKeyboard(unsigned char key, int x, int y){
  (myZprInstances->at(this->getActiveZprInstanceID()))->keyboard(key,x,y);
}

void zprManager::zprKeyboardUp(unsigned char key, int x, int y){
  (myZprInstances->at(this->getActiveZprInstanceID()))->keyboardUp(key,x,y);
}

void zprManager::zprSpecial(int key, int x, int y){
  (myZprInstances->at(this->getActiveZprInstanceID()))->special(key,x,y);
}

void zprManager::zprSpecialUp(int key, int x, int y){
  (myZprInstances->at(this->getActiveZprInstanceID()))->specialUp(key,x,y);
}

void zprManager::zprDisplay(){
  if(forceUpdate){
    int bb;
    for(bb = 0; bb < this->nextZprInstanceID; bb++){
      if((*myZprInstances)[bb]->forceUpdate){
        (*myZprInstances)[bb]->drawGraphicsExternal(); // this excludes the current instance
      }
    }
  }
  // plot the current instance.
  (myZprInstances->at(this->getActiveZprInstanceID()))->display();//keyboard(key,x,y);
}

void zprManager::zprIdle(){
  (myZprInstances->at(this->getActiveZprInstanceID()))->idle();
  usleep(5000);
}

void zprManager::zprDrawGraphics(){
  (myZprInstances->at(this->getActiveZprInstanceID()))->drawGraphics();//keyboard(key,x,y);
}

void zprInstance::drawGraphics(){
  // printf("zprInstance::drawGraphics() id=%d\n", myZprInstanceID);
  std::vector<glPlottable *>::iterator it;
  int i = 0;
  for(it = myGraphics.begin(); it!=myGraphics.end(); it++){
    // cout << "\tzprInstance::drawGraphics() id=" << myZprInstanceID << " i = " << i << " of " << myGraphics.size() << " type: " << (*it)->myType << endl;
    (*it)->drawMe();
    /*
    if(false){
      // (*it)->myType != NULL)
      if( (*it)->myType.compare(std::string("glMusicSphere")) == 0){
        //reflexive (yin yang inside out) fxn.
      }
    }
    */
    if((*it)->forceUpdate){
      forceUpdate = true;
      myZprManager->forceUpdate = true;
    }
    i++;
  }
  // printf("drawGraphics() return\n");
}

void zprInstance::drawGraphicsExternal(){
  int zprID = myZprManager->getActiveZprInstanceID();
  int activeGLUTID = glutGetWindow();
  if(myGlutID() == activeGLUTID) return;
  glutSetWindow(myGlutID());
  dprintf("Draw graphics external.. Caller ID (%d) My ID (%d)", activeGLUTID, myGlutID());

  GLERROR;
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  std::vector<glPlottable *>::iterator it;
  int i=0;
  mark();
  for(it = myGraphics.begin(); it!=myGraphics.end(); it++){
    ++i;
    if((*it)->isLinkd){
      dprintf(" Drawing LINKED item (%d of %d) \n",i,myGraphics.size());
      (*it)->drawMe(true);
    }
    else{
      (*it)->drawMe();
    }
  }
  glutSwapBuffers();
  GLERROR;
  glutSetWindow(activeGLUTID);
  dprintf("Return from drawGraphics()");
}

void zprInstance::display(){
  if(isPaused) return;
  if(!refreshflag) return;
  GLERROR;
  if(glutGetWindow() != myGlutID()){
    cout <<"Error: display() was called on nonactive window myID()=("<<myGlutID()<<") glutGetWindow=(" <<glutGetWindow()<<endl;
    exit(1);
  }
  glutSetWindow(myGlutID()); //obviously redundant.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // printf("display::drawGraphics()\n");
  this->drawGraphics();
  // printf("display::drawText()\n");
  this->drawText();
  glutSwapBuffers();
  GLERROR;
  if(!forceUpdate){
    refreshflag = false;
  }
  else{
  }
  // printf("return\n");
}

void zprInstance::idle(){
  if(refreshflag){
    glFlush();
    glutPostRedisplay();
  }
}

int zprInstance::grabint(char * p){
  return atoi(p);
}

void zprInstance::setrgb(int r, int g, int b){
  myBi->at(0) = r;
  myBi->at(1) = g;
  myBi->at(2) = b;

  // gotta have the trickle-down. N.b. the glImage()::rebuffer() gets the
  // band-select info from zprInstance
  for(vector<glPlottable *>::iterator it = myGraphics.begin(); it != myGraphics.end(); it++){
    if((*it)->myType.compare(std::string("glImage")) == 0){
      // cout << "\tmyGraphics " << (*it)->myType << " rebuffer " << endl;
      ((glImage *)((void *)((glPlottable *)(*it))))->rebuffer();
    }
  }
}

void zprInstance::getrgb(int & r, int & g, int & b){
  r = myBi->at(0);
  g = myBi->at(1);
  b = myBi->at(2);
}

void zprInstance::processString(){

  char strSleep[] = "sleep";
  int i = 0;
  if(console_string[i] == '\0'){
    isPaused = !isPaused;
    return;
  }
  while(console_string[i] != '\0'){
    i++;
  }
  int count = i + 1;
  SA<char> s(count);

  strcpy(&s[0], console_string);
  int r,g,b, tk;
  int ndim = NBand;
  if(strcmpz(&s[0], &strSleep[0])){
    int st = atoi(&s[5]);
    if(st < 1){
      return;
    }
    usleepNupdate = st;
    return;
  }
  i = grabint(&s[1]);
  switch(s[0]){
    case 'd':
    resetViewPoint();
    break;

    case 'c':
    printf("resetView();\n");
    refreshflag = true;
    mark();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(_left,_right,_bottom,_top,_zNear,_zFar);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    resetView();
    display();
    return;
    break;

    case 'r':
    if(i < 1 || i > ndim) break;
    getrgb( r,g,b);
    setrgb(i - 1, g, b);
    break;
    case 'g':
    if( (i<1) || (i>ndim)) break;
    getrgb(r, g, b);
    setrgb(r, i - 1, b);
    break;
    case 'b':
    if(i < 1 || i > ndim) break;
    getrgb(r, g, b);
    setrgb(r, g, i - 1);
    break;

    case 'p':
    getrgb(r, g, b);
    printf("r band %d g band %d b band %d\n", r + 1, g + 1, b + 1);

    break;
    default:

    try{
      int i = atoi(&s[0]);
      if(i >= 1 && i <= ndim){
        setrgb(i - 1 , i - 1, i - 1);
      }

    }
    catch(std::exception e){

    }

    break;
  }
}

void zprInstance::special(int key, int x, int y){
  refreshflag = true; //need to draw graphics.
  if(key == GLUT_KEY_F1){
    _F1 = true;
    printf("Move atom active\n");
    //printf("Special Key dn: %d, %d\n", key, _F1);
  }
  if(key == GLUT_KEY_F2){
    _F2 = !_F2;
    printf("groundref class colouring toggle\n");
  }
}

void zprInstance::specialUp(int key, int x, int y){
  refreshflag = true;
  if( key ==GLUT_KEY_F1){
    _F1 = false;
  }
  /* 
  if( key ==GLUT_KEY_F2){
    _F2 = false;
  }
  */
  // should actually have a toggle vs. on/ off mode function available but whatever
}

void zprInstance::keyboardUp(unsigned char key, int x, int y){
  refreshflag = true; //need to draw graphics.
  // printf("Key up: %d\n", key);
  switch(key){
    default:
    break;
  }
}

void zprInstance::keyboard(unsigned char key, int x, int y){
  printf("Key down: %d\n", key);
  refreshflag = true; //need to draw graphics.
  switch(key){
    case 127:
    case 8:

    if(console_position > 0){
      console_position --;
      console_string[console_position]='\0';
      display();
    }
    break;
    //_____________________________________________________________________________
    //_____________________________________________________________________________
    //--> ENTER <--
    //executes Acommand. ==========================================================
    case 13 :
    //printf( "%d Pressed RETURN\n",(char)key);
    processString();
    console_string[0]='\0';
    console_position=0;
    //mark(): need to rebuffer / update any subordinate "glPlottable" graphics objects.
    mark();
    display();
    break;
    //_____________________________________________________________________________
    //_____________________________________________________________________________

    // Escape
    case 27 :
    quitme();
    exit(0);
    //printf( "%d Pressed Esc\n",(char)key);
    break;

    // Delete
    /* case 127 :
    printf( "%d Pressed Del\n",(char)key);
    break;
    */
    default:
    //printf( "Pressed key %c AKA %d at position %d % d\n",(char)key, key, x, y);
    console_string[console_position++] = (char)key;
    console_string[console_position]='\0';
    // printf("STRING: %s\n", &console_string[0]);
    display();
    break;
  }
}

void zprInstance::glutGetWindowPosition( int & x, int & y){
  focus();
  x = glutGet(GLUT_WINDOW_X);
  y = glutGet(GLUT_WINDOW_Y);
}

int zprInstance::setScreenPositionAndReturnBorderHeight(int x, int y){
  focus();
  glutPositionWindow(x,y);
  myScreenPositionX = x;
  myScreenPositionY = y;
  int newX, newY;
  glutGetWindowPosition( newX, newY);
  return(0);
  int borderHeight = y-newY;//newY - y;
  myScreenPositionX = newX;
  myScreenPositionY = newY;
  return(borderHeight);
}
void zprInstance::setScreenPosition(int x, int y){
  int borderHeight = setScreenPositionAndReturnBorderHeight(x,y);
}
void zprInstance::setRightOf( zprInstance * other){
  focus();
  setScreenPosition(
  other->myScreenPositionX + other->myWindowWidth,
  other->myScreenPositionY
  );
}
void zprInstance::setBelow(zprInstance * other){
  focus();
  setScreenPosition(
  other->myScreenPositionX,
  other->myScreenPositionY+other->myWindowHeight // +67
  );
}

void zprInstance::zprInit(){
  zprReferencePoint[0] = 0.;
  zprReferencePoint[1] = 0.;
  zprReferencePoint[2] = 0.;
  zprReferencePoint[3] = 0.;
  getMatrix();
  glutReshapeFunc( myZprReshape );
  glutMouseFunc ( myZprMouse );
  glutMotionFunc ( myZprMotion );
  char asdf[1000]="\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
  sprintf(&asdf[0], "GLUTID (%d)\0", myGlutID());
  glutSetWindowTitle(&(asdf[0]));

}
void zprInstance::setTitle(string t){
  focus();
  glutSetWindowTitle(t.c_str());
}

void zprInstance::zprReshape(int w,int h){

  GLfloat ratio; // http://faculty.ycp.edu/~dbabcock/cs370/labs/lab07.html

  // Set new screen extents
  glViewport(0,0,w,h);

  // Select projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Adjust viewing volume (orthographic)
  // If taller than wide adjust y
  if(w <= h){
    ratio = (GLfloat) h/ (GLfloat) w;
    glOrtho(-1.0f,1.0f,-1.0f*ratio,1.0f*ratio,-1.0f,1.0f);
    _bottom = -1.*ratio; _top = 1.*ratio;
  }
  // If wider than tall adjust x
  else if (h <= w){
    ratio = (GLfloat) w/ (GLfloat) h;
    glOrtho(-1.0f*ratio,1.0f*ratio,-1.0f,1.0f,-1.0f,1.0f);
    _left = -1.*ratio; _right = 1.*ratio;
  }
  //glPopMatrix();
  //glMatrixMode(GL_MODELVIEW);
  myWindowWidth = glutGet( GLUT_WINDOW_WIDTH );
  myWindowHeight = glutGet( GLUT_WINDOW_HEIGHT );
  refreshflag = true;
  glMatrixMode(GL_MODELVIEW);

  return;

}
//http://graphics.stanford.edu/courses/cs248-01/OpenGLHelpSession/code_example.html

void zprInstance::zprMouse(int button, int state, int x, int y){
  printf("zprMouse()\n");
  GLint viewport[4]; /* Do picking */
  refreshflag = true;

  if (state==GLUT_DOWN && button==GLUT_LEFT_BUTTON){
    zprPick(x,glutGet(GLUT_WINDOW_HEIGHT)-1-y,3,3);
    _pickme(0);
  }
  else{
    pick(-1);
  }

  _mouseX = x;
  _mouseY = y;

  if (state==GLUT_UP){
    switch (button){
      case GLUT_LEFT_BUTTON: _mouseLeft = false; myPickNames.clear(); break;
      case GLUT_MIDDLE_BUTTON: _mouseMiddle = false; break;
      case GLUT_RIGHT_BUTTON: _mouseRight = false; break;
    }
  }
  else{
    switch (button){
      case GLUT_LEFT_BUTTON: _mouseLeft = true; break;
      case GLUT_MIDDLE_BUTTON: _mouseMiddle = true; break;
      case GLUT_RIGHT_BUTTON: _mouseRight = true; break;
    }
  }

  if(ZPR_ZOOM_MODE || ZPR_PAN_MODE || ZPR_ROTATE_MODE){
    _mouseLeft = _mouseMiddle = _mouseRight = false;
    _mouseLeft = ZPR_ROTATE_MODE;
    _mouseRight = ZPR_PAN_MODE;
    _mouseMiddle = ZPR_ZOOM_MODE;
  }

  printf("\nmouse (x/col,y/row)=(%d,%d) myZprInstanceID=%d\n", x, y, myZprInstanceID);

  if(myZprInstanceID == 1){
    // subset window
    WIN_I = (y + NWIN) >= SUB_MM ? SUB_MM - NWIN : y;
    WIN_J = (x + NWIN) >= SUB_MM ? SUB_MM - NWIN : x;

    SA<float> * dat4 = TGT;
    size_t i, j, k;

    // print out data under cursor
    printf("bi\tbn\td\n");
    for0(k, IMG_NB){
      for0(i, 1){
        for0(j, 1){
          float d = (*SUB)[(k * SUB_MM * SUB_MM) + (SUB_MM * (WIN_I + i)) + (WIN_J + j)];
          printf("%d\t%s\t%e\n", k, vec_band_names[k].c_str(), (double)d);
        }
      }
    }

    // do the work
    for0(k, IMG_NB){
      for0(i, NWIN){
        for0(j, NWIN){
          float d = (*SUB)[(k * SUB_MM * SUB_MM) + (SUB_MM * (WIN_I + i)) + (WIN_J + j)];
          (*dat4)[(k * NWIN* NWIN) + (i * NWIN) + j] = d;
        }
      }
    }
    TGT_MYIMG->initFrom(dat4, NWIN, NWIN, IMG_NB);
    ((glImage *)TGT_GLIMG)->rebuffer();
    zprInstance * p = this;
    zprInstance * a = myZprManager->myZprInstances->at(2);
    a->focus();
    a->mark();
    a->display();

    focus();
    mark();
    display();
    myZprDisplay();
  }

  if(myZprInstanceID == 0){
    // overview window
    printf("SUB_SCALE_F %f\n", SUB_SCALE_F);

    size_t dx = (int)floor(((float)x) / SUB_SCALE_F);
    size_t dy = (int)floor(((float)y) / SUB_SCALE_F);
    printf("dx %zu dy %zu\n", dx, dy);
    printf("(IMG_NR - SUB_MM) %zu (IMG_NC - SUB_MM) %zu\n", (IMG_NR - SUB_MM), (IMG_NC - SUB_MM));

    // overflow protect
    if(dy >= (IMG_NR - SUB_MM)){
      dy = IMG_NR - SUB_MM;
    }
    if(dx >= (IMG_NC - SUB_MM)){
      dx = IMG_NC - SUB_MM;
    }

    SUB_START_J = dx;
    SUB_START_I = dy;

    printf("IMG_NR %zu IMG_NC %zu\n", IMG_NR, IMG_NC);
    printf("dx %zu dy %zu\n", dx, dy);

    // big-data resilient read!
    SA<float> * dat3 = SUB;
    if(true){
      load_sub_np = IMG_NR * IMG_NC;
      load_sub_nb = IMG_NB;
      load_sub_mm = SUB_MM;
      load_sub_i_start = SUB_START_I; //i_start;
      load_sub_j_start = SUB_START_J; //j_start;
      load_sub_nc = IMG_NC; //nc;
      load_sub_dat3 = &((*dat3)[0]);
      load_sub_infile = IMG_FN; //string(infile);
      parfor(0, IMG_NB, load_sub);
    }

    SUB_MYIMG->initFrom(dat3, SUB_MM, SUB_MM, IMG_NB);
    ((glImage *)SUB_GLIMG)->rebuffer();
    for(int m = 0; m < myZprManager->myZprInstances->size(); m++){
      if(m > 1) continue; // update the first two windows (otherwise get segfault)
      zprInstance * a = myZprManager->myZprInstances->at(m);
      a->focus();
      a->mark();
      a->display();
    }
    myZprDisplay();
  }

  if(myZprInstanceID == 0){

    std::vector<glPlottable *>::iterator it;
    int i = 0;
    it = myGraphics.begin();
    glImage * image = (glImage * )(glPlottable *) (*it);
    int NRow = image->image->NRow;
    int NCol = image->image->NCol;
    int NBand = image->image->NBand;
    SA< SA<float> * > * FB = image->image->getFloatBuffers();
    for(i = 0; i < NBand; i++){
      // could turn this on to print image information
      // cout << "\tband " << i << "->" << FB->at(i)->at(y * NCol + x) << endl;
    }
  }

  if(myZprInstanceID < 3){
    std::vector<glPlottable *>::iterator it;
    int i = 0;
    it = myGraphics.begin();
    glImage * image = (glImage * )(glPlottable *) (*it);
    int NRow = image->image->NRow;
    int NCol = image->image->NCol;
    int NBand = image->image->NBand;
    SA< SA<float> * > * FB = image->image->getFloatBuffers();
    printf("bi\tbn\td\n");
    for(i = 0; i < NBand; i++){
      // could turn this on to print image information
      printf("%d\t%s\t%e\n", i, vec_band_names[i].c_str(), (double)(FB->at(i)->at(y * NCol + x)));
    }
  }

  // update the scatter plot window
  zprInstance * s = myZprManager->myZprInstances->at(3);
  s->focus();
  s->mark();
  s->display();

  if(callMeWithMouseClickCoordinates){
    (*callMeWithMouseClickCoordinates)(x,y);
  }
  glGetIntegerv(GL_VIEWPORT,viewport);
  pos(&_dragPosX,&_dragPosY,&_dragPosZ,x,y,viewport);
  glutPostRedisplay();
}

void zprInstance::zprMotion(int x, int y){
  bool changed = false;
  //refreshflag = false;
  const int dx = x - _mouseX;
  const int dy = y - _mouseY;
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT,viewport);
  double px,py,pz; //So far, missing these in the newzpr.cpp code (12:15am, 2013 December 19th)
  pos(&px,&py,&pz,x,y,viewport);

  //this->getMatrix();
  if(_F1 ){
    if( _mouseLeft){
      if(myPickNames.size()<1){
        return;
      }
      //cout << "F1 active PickSet:";

      //Move atom(s). (inspired by Latest2.tar.gz: MoleculeCanvas.cpp: MouseMoveAtoms
      //should evolve this method to use a "selected item" a-la Latest-2 syncitium.
      std::set<GLint>::iterator it;
      for(it=myPickNames.begin(); it!=myPickNames.end(); it++){
        //std::map<GLint, glPlottable* > myGraphicsAsAFunctionOfGLName;
        if( myGraphicsAsAFunctionOfGLName.count( *it) != 1) continue; //don't move an object unless we have a unique reference (based on this name) of an object to move.
        glPlottable * a = myGraphicsAsAFunctionOfGLName.find( *it )->second;
        a->setRelativePosition = 0;
        GLdouble proj[16]; // vars
        GLdouble model[16];
        GLint view[4];
        GLdouble nearx,neary,nearz;

        glGetDoublev(GL_PROJECTION_MATRIX,proj); // get projection, model, and view matrixes
        glGetDoublev(GL_MODELVIEW_MATRIX,model);
        glGetIntegerv(GL_VIEWPORT,view);

        float screendx=(float)x-(float)_mouseX;
        float screendy=(float)y-(float)_mouseY;
        //cout << *it << "," <<screendx <<","<<screendy<<"..";
        double mx, my, mz, vx, vy, vz;
        mx = (double) a->x.x;//_GetX();
        my = (double) a->x.y;//GetY();
        mz = (double) a->x.z;//GetZ();
        //world xyz onto screen xyz
        gluProject(mx,my,mz,model,proj,view,&vx,&vy,&vz);

        float screeny = vy - screendy;//- (float)screeny;
        float screenx = vx + screendx ;

        //screen xyz onto world xyz
        gluUnProject(screenx, screeny, vz, model,proj,view,&nearx,&neary,&nearz);
        //printf("{%f,%f,%f}\n", nearx, neary, nearz);
        vec3 newa((float)nearx - mx, (float)neary -my, (float)nearz-mz); //for(int k=0; k < GetNumberSelectedItems(); k++) {
          //SYNAtom* ak = GetSelectedItem(k)->getAtom();
          vec3 translation(
          ((double) a->x.x) +newa.x,
          ((double) a->x.y) +newa.y,
          ((double) a->x.z) +newa.z
          );
          a->x.x = translation.x;a->x.y = translation.y; a->x.z = translation.z; // = translation;*/

          a->Update = true; glutPostRedisplay(); display(); glutSwapBuffers(); //glutPostRedisplay(); display();//glutSwapBuffers();//refresh();
          //glutPostRedisplay();
          _dragPosX = px;
          _dragPosY = py;
          _dragPosZ = pz;

          _mouseX = x;
          _mouseY = y;

          changed = true; refreshflag = true;
        }
        //end Move atom.
      }
    }
    else{
      if (_mouseMiddle || (_mouseLeft && _mouseRight)){
        //ZOOM
        double s = exp((double)dy*0.01);
        glTranslatef( zprReferencePoint[0], zprReferencePoint[1], zprReferencePoint[2]);
        glScalef(s,s,s);
        glTranslatef(-zprReferencePoint[0],-zprReferencePoint[1],-zprReferencePoint[2]);
        changed = true; refreshflag = true;
      }
      else
      if (_mouseLeft){
        //ROTATE
        double ax,ay,az,bx,by,bz,angle;
        ax = dy;
        ay = dx;
        az = 0.0;
        angle = vlen(ax,ay,az)/(double)(viewport[2]+1)*180.0;
        /* Use inverse matrix to determine local axis of rotation */
        bx = _matrixInverse[0]*ax + _matrixInverse[4]*ay + _matrixInverse[8] *az;
        by = _matrixInverse[1]*ax + _matrixInverse[5]*ay + _matrixInverse[9] *az;
        bz = _matrixInverse[2]*ax + _matrixInverse[6]*ay + _matrixInverse[10]*az;
        glTranslatef( zprReferencePoint[0], zprReferencePoint[1], zprReferencePoint[2]);
        glRotatef(angle,bx,by,bz);
        glTranslatef(-zprReferencePoint[0],-zprReferencePoint[1],-zprReferencePoint[2]);
        refreshflag = true;
        changed = true;
      }
      else
      if(_mouseRight){
        //PAN
        //printf("PANNING..........\n");
        double px,py,pz;
        glMatrixMode(GL_MODELVIEW);
        pos(&px,&py,&pz,x,y,viewport);
        //glPushMatrix();
        glLoadIdentity();
        glTranslatef(px-_dragPosX,py-_dragPosY,pz-_dragPosZ);
        glMultMatrixd(_matrix);
        _dragPosX = px; _dragPosY = py; _dragPosZ = pz;
        changed = true; refreshflag = true;
        //glPopMatrix();
        //glutPostRedisplay();
        //return;
      }
      _mouseX = x; _mouseY = y;
    }
    if (changed){
      refreshflag = true;
      this->getMatrix();
      glutPostRedisplay();
    }
    // changed = false;
    // refreshflag = false;
  }

  void zprInstance::_pickme(GLint name){
    if( _mouseMiddle || _mouseRight ) return;

    refreshflag = true;
    //set<int> * myPickNames = &(athis->myPickNames);
  if(myPickNames.size()<1){ return;}
  cout << "PickSet:";
  std::set<GLint>::iterator it;
  for(it=myPickNames.begin(); it!=myPickNames.end(); it++){
    cout << *it << "," ;
  }
  cout <<"PickLabelSet: ";
  std::vector<glPlottable *>::iterator ti;
  for(ti = myGraphicsLabelled.begin(); ti!=myGraphicsLabelled.end(); ti++){
    if( (*ti)->isPicked() && (*ti)->isLabelled ){
      int pickedLabel = (*ti)->myLabel;
      cout << pickedLabel<<",";
      //Update = true;
      myZprManager->mark();
      break;
    }
  }
  //for(it=myLabelsPicked->begin(); it!=myLabelsPicked->end(); it++)
  // cout << *it << ",";
  //
  cout << endl;
  fflush(stdout);
}

void zprInstance::renderBitmapString(float x, float y, void *font, char *string){
  char *c;
  glRasterPos2f(x,y);
  for (c=string; *c != '\0'; c++){
    glutBitmapCharacter(font, *c);
  }
}

//http://www.codeproject.com/Articles/80923/The-OpenGL-and-GLUT-A-Powerful-Graphics-Library-an
void zprInstance::setOrthographicProjection() {
  int h = this->myWindowHeight;//>WINDOWY;
  int w = this->myWindowWidth;//WINDOWX;
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0., (float)w, 0., (float)h);
  glScalef(1., -1., 1.);
  glTranslatef(0, -1.*(float)h, 0);
  glMatrixMode(GL_MODELVIEW);
}

void zprInstance::resetPerspectiveProjection(){
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void zprInstance::drawText(){
  glColor3f(0.0f,1.0f,0.0f);
  setOrthographicProjection();
  glPushMatrix();
  glLoadIdentity();
  int lightingState = glIsEnabled(GL_LIGHTING);
  if(lightingState) glDisable(GL_LIGHTING);
  renderBitmapString(3,(this->myWindowHeight)-3,(void *)MYFONT,this->console_string);
  if(lightingState) glEnable(GL_LIGHTING);
  glPopMatrix();
  resetPerspectiveProjection();
}

void zprInstance::add(glPlottable * a){
  //zprInstance member:
  //std::map<int, glPlottable* >myGraphicsAsAFunctionOfGLName;
  std::map<GLint, glPlottable*>::iterator it = myGraphicsAsAFunctionOfGLName.begin();
  myGraphicsAsAFunctionOfGLName.insert(it, std::pair<GLint, glPlottable*>( a->myName, a));
  myGraphics.push_back(a);
  if(a->isLabelled){
    myGraphicsLabelled.push_back(a);
  }
}

/*****************************************************************
* Utility functions
*****************************************************************/

double zprInstance::vlen(double x,double y,double z){
  return sqrt(x*x+y*y+z*z);
}

void zprInstance::pos(double *px,double *py,double *pz,const int x,const int y,const int *viewport){
  /*
  Use the ortho projection and viewport information
  to map from mouse co-ordinates back into world
  co-ordinates
  */

  *px = (double)(x-viewport[0])/(double)(viewport[2]);
  *py = (double)(y-viewport[1])/(double)(viewport[3]);

  *px = _left + (*px)*(_right-_left);
  *py = _top + (*py)*(_bottom-_top);
  *pz = _zNear;
}

void zprInstance::getMatrix(){
  glGetDoublev(GL_MODELVIEW_MATRIX,this->_matrix);
  this->invertMatrix(this->_matrix,this->_matrixInverse);
}

/*
* From Mesa-2.2\src\glu\project.c
*
* Compute the inverse of a 4x4 matrix. Contributed by scotter@lafn.org
*/

void zprInstance::invertMatrix(const GLdouble *m, GLdouble *out ){
  /* NB. OpenGL Matrices are COLUMN major. */
  #define MAT(m,r,c) (m)[(c)*4+(r)]

  /* Here's some shorthand converting standard (row,column) to index. */
  #define m11 MAT(m,0,0)
  #define m12 MAT(m,0,1)
  #define m13 MAT(m,0,2)
  #define m14 MAT(m,0,3)
  #define m21 MAT(m,1,0)
  #define m22 MAT(m,1,1)
  #define m23 MAT(m,1,2)
  #define m24 MAT(m,1,3)
  #define m31 MAT(m,2,0)
  #define m32 MAT(m,2,1)
  #define m33 MAT(m,2,2)
  #define m34 MAT(m,2,3)
  #define m41 MAT(m,3,0)
  #define m42 MAT(m,3,1)
  #define m43 MAT(m,3,2)
  #define m44 MAT(m,3,3)

  GLdouble det;
  GLdouble d12, d13, d23, d24, d34, d41;
  GLdouble tmp[16]; /* Allow out == in. */

  /* Inverse = adjoint / det. (See linear algebra texts.)*/

  /* pre-compute 2x2 dets for last two rows when computing */
  /* cofactors of first two rows. */
  d12 = (m31*m42-m41*m32);
  d13 = (m31*m43-m41*m33);
  d23 = (m32*m43-m42*m33);
  d24 = (m32*m44-m42*m34);
  d34 = (m33*m44-m43*m34);
  d41 = (m34*m41-m44*m31);

  tmp[0] = (m22 * d34 - m23 * d24 + m24 * d23);
  tmp[1] = -(m21 * d34 + m23 * d41 + m24 * d13);
  tmp[2] = (m21 * d24 + m22 * d41 + m24 * d12);
  tmp[3] = -(m21 * d23 - m22 * d13 + m23 * d12);

  /* Compute determinant as early as possible using these cofactors. */
  det = m11 * tmp[0] + m12 * tmp[1] + m13 * tmp[2] + m14 * tmp[3];

  /* Run singularity test. */
  if (det == 0.0) {
    /* printf("invert_matrix: Warning: Singular matrix.\n"); */
    /* memcpy(out,_identity,16*sizeof(double)); */
  }
  else {
    GLdouble invDet = 1.0 / det;
    /* Compute rest of inverse. */
    tmp[0] *= invDet;
    tmp[1] *= invDet;
    tmp[2] *= invDet;
    tmp[3] *= invDet;

    tmp[4] = -(m12 * d34 - m13 * d24 + m14 * d23) * invDet;
    tmp[5] = (m11 * d34 + m13 * d41 + m14 * d13) * invDet;
    tmp[6] = -(m11 * d24 + m12 * d41 + m14 * d12) * invDet;
    tmp[7] = (m11 * d23 - m12 * d13 + m13 * d12) * invDet;

    /* Pre-compute 2x2 dets for first two rows when computing */
    /* cofactors of last two rows. */
    d12 = m11*m22-m21*m12;
    d13 = m11*m23-m21*m13;
    d23 = m12*m23-m22*m13;
    d24 = m12*m24-m22*m14;
    d34 = m13*m24-m23*m14;
    d41 = m14*m21-m24*m11;

    tmp[8] = (m42 * d34 - m43 * d24 + m44 * d23) * invDet;
    tmp[9] = -(m41 * d34 + m43 * d41 + m44 * d13) * invDet;
    tmp[10] = (m41 * d24 + m42 * d41 + m44 * d12) * invDet;
    tmp[11] = -(m41 * d23 - m42 * d13 + m43 * d12) * invDet;
    tmp[12] = -(m32 * d34 - m33 * d24 + m34 * d23) * invDet;
    tmp[13] = (m31 * d34 + m33 * d41 + m34 * d13) * invDet;
    tmp[14] = -(m31 * d24 + m32 * d41 + m34 * d12) * invDet;
    tmp[15] = (m31 * d23 - m32 * d13 + m33 * d12) * invDet;

    memcpy(out, tmp, 16*sizeof(GLdouble));
  }

  #undef m11
  #undef m12
  #undef m13
  #undef m14
  #undef m21
  #undef m22
  #undef m23
  #undef m24
  #undef m31
  #undef m32
  #undef m33
  #undef m34
  #undef m41
  #undef m42
  #undef m43
  #undef m44
  #undef MAT
}

/***************************************** Picking ****************************************************/

void zprInstance::zprSelectionFunc(void (*f)(void)){
  this->selection = f;
}

void zprInstance::zprPickFunc(void (*f)(GLint name)){
  this->pick = f;
}

/* Draw in selection mode */

void zprInstance::processHits(GLint hits, GLuint buffer[]){
  //std::set<GLint> myPickNames;
  this->myPickNames.clear();

  unsigned int i, j;
  GLuint names, *ptr, minZ,*ptrNames, numberOfNames;

  printf ("hits = %d names:{", hits);
  ptr = (GLuint *) buffer;
  minZ = 0xffffffff;
  for (i = 0; i < hits; i++) {
    //printf("(i%d)",i);
    names = *ptr;
    ptr++;
    GLint mindepth = *ptr; ptr++;
    GLint maxdepth = *ptr; ptr++;
    for(j=0; j<names; j++){
      GLint name = *ptr;
      printf(",%d",name);
      if(name>=0){
        this->myPickNames.insert(name);
      }
      ptr++;
    }
  }
  printf ("}\n");
}

void zprInstance::zprPick(GLdouble x, GLdouble y,GLdouble delX, GLdouble delY){
  if( _mouseMiddle || _mouseRight ) return;
  GLuint buffer[MAXBUFFERSIZE];
  const int bufferSize = sizeof(buffer)/sizeof(GLuint);

  GLint viewport[4];
  GLdouble projection[16];

  GLint hits;
  GLint i,j,k;

  GLint min = -1;
  GLuint minZ = -1;

  glSelectBuffer(bufferSize,buffer); /* Selection buffer for hit records */
  glRenderMode(GL_SELECT); /* OpenGL selection mode */
  glInitNames(); /* Clear OpenGL name stack */

  glMatrixMode(GL_PROJECTION);
  glPushMatrix(); /* Push current projection matrix */
  glGetIntegerv(GL_VIEWPORT,viewport); /* Get the current viewport size */
  glGetDoublev(GL_PROJECTION_MATRIX,projection); /* Get the projection matrix */
  glLoadIdentity(); /* Reset the projection matrix */
  gluPickMatrix(x,y,delX,delY,viewport); /* Set the picking matrix */
  glMultMatrixd(projection); /* Apply projection matrix */

  glMatrixMode(GL_MODELVIEW);

  if(this->selection){
    this->selection();
  }
  // Draw the scene in selection mode
  hits = glRenderMode(GL_RENDER); /* Return to normal rendering mode */
  this->processHits( hits, buffer);

  glMatrixMode(GL_PROJECTION);
  glPopMatrix(); /* Restore projection matrix */
  glMatrixMode(GL_MODELVIEW);
  return;
}
void glPlottable::initName(zprInstance * parent, int useName){
  isInverted = false;
  parentZprInstance = parent;
  myName = -1;
  if(useName){
    myName = parentZprInstance->nextGLUTName();//myPickNames.count(myName));
    parentZprInstance->advanceGLUTName();
  }
  parentZprInstance->add( (glPlottable *) this);
  return;
}

void glPoints::drawMe(){
  printf("glPoints::drawMe()\n");
  size_t nr = myI->image->NRow;
  size_t nc = myI->image->NCol;
  size_t nf = nr * nc * 3;
  float * d = myI->dat->elements;
  float r, g, b, h, s, v;
  float class_label;

  glColor3f(1,1,1);
  if(!myI->myParent->_F2){
    for(size_t i = 0; i < nf; i += 3){
      r = d[i];
      g = d[i+1];
      b = d[i+2];
      glColor3f(r, g, b);
      glBegin(GL_POINTS);
      glVertex3f(r, g, b);
      glEnd();
      // printf("%f %f %f]\n", d[i], d[i+1], d[i+2]);
    }
  }
  else{
    s = v = 1.;
    d = myI->class_label->elements;
    float * dd = myI->dat->elements;
    for(size_t i = 0; i < nf; i += 3){
      class_label = d[i / 3];
      if(class_label == 0.){
        r = g = b = 0.;
      }
      else if(class_label == -1.){
        r = g = b = 1.;
      }
      else{
        h = 360. * (float)(class_label - 1) / (float)(groundref.size());
	if(h > 360.){
		printf("class_label %f groundref.size() %zu\n", class_label, (size_t)groundref.size());
		err("out of range");
	}

        hsv_to_rgb(&r, &g, &b, h, s, v);
      }

      glColor3f(r, g, b);
      glBegin(GL_POINTS);
      glVertex3f(dd[i], dd[i + 1], dd[i + 2]);
      glEnd();
    }
   
    for(size_t i = 0; i < groundref.size(); i++){
	const char * class_string = vec_band_names[groundref[i]].c_str();
	h = 360. * (float)(i) / (float)(groundref.size());
        hsv_to_rgb(&r, &g, &b, h, s, v);

    	glColor3f(r, g, b);
    	parentZprInstance->setOrthographicProjection();
    	glPushMatrix();
    	glLoadIdentity();
    	int lightingState = glIsEnabled(GL_LIGHTING);
    	if(lightingState) glDisable(GL_LIGHTING);
	int dy = (parentZprInstance->myWindowHeight)- 3 - (MYFONT_SIZE * i);
    	parentZprInstance->renderBitmapString(3,dy,(void *)MYFONT,(char *) (void*)class_string);
    	if(lightingState) glEnable(GL_LIGHTING);
    	glPopMatrix();
    	parentZprInstance->resetPerspectiveProjection();
    }
  }
}
