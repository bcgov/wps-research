/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere.. */
#include <math.h>
#include "newzpr.h"
#include <stdlib.h>
#include <memory.h>
#include <iostream>
using namespace std;

myImg * SCENE_MYIMG = NULL;
void * SCENE_GLIMG = NULL;
map<SA<float> *, float> scene_band_min;
map<SA<float> *, float> scene_band_max;
map<set<SA<float> *>, float> scene_bands_min;
map<set<SA<float> *>, float> scene_bands_max;

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

// target / analysis window magnification parameter
float TGT_MAGNIFICATION = 10.;

int USE_PROPORTIONAL_SCALING = true; // default to proportional scaling
// groundref detection
vector<int> groundref;
vector<string> vec_band_names;
set<int> groundref_disable;

// other
int bands_per_frame; // bands per frame if known, should autodetect this

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

// declare a map (SA<float> * b, to min/max).. no recalc!
void two_percent(float & min, float & max, SA<float> * b){
  if(scene_band_min.count(b) < 1){
    // not actually 2%, gasp! the real deal should calculate on intensity..
    priority_queue<float> q;
    float * d = b->elements;
    unsigned int n_two = floor(0.02 * ((float)b->size()));
    unsigned int i;
    for(i = 0; i < b->size(); i++) q.push(d[i]);
    for(i = 0; i < n_two; i++) q.pop();
    max = q.top();
    while(q.size() > n_two) q.pop();
    min = q.top();
    //while(q.size() > 0) q.pop();
    scene_band_min[b] = min;
    scene_band_max[b] = max;
  }
  else{
    min = scene_band_min[b];
    max = scene_band_max[b];
  }
  printf("two_p n=%zu min %f max %f\n", b->size(), min, max);
}

inline float max3(float r, float g, float b){
  return max(r, max(g, b));
}

void two_percent(float & min, float & max, SA<float> * r, SA<float> * g, SA<float> * b){
  // compared with the above, this version implements a "proportional" scaling
  set<SA<float> *> bs;
  bs.insert(r);
  bs.insert(g);
  bs.insert(b); // build the tuple to see if we've already calculated this

  if(scene_bands_min.count(bs) < 1){

    priority_queue<float> q;
    float * R= b->elements;
    float * G = g->elements;
    float * B = b->elements;

    unsigned int n_two = floor(0.02 * ((float)b->size()));
    unsigned int i;
    for(i = 0; i < b->size(); i++) q.push(max3(R[i], G[i], B[i]));
    for(i = 0; i < n_two; i++) q.pop();
    max = q.top();
    while(q.size() > n_two) q.pop();
    min = q.top();
    // while(q.size() > 0) q.pop();
  }
  else{
    min = scene_bands_min[bs];
    max = scene_bands_max[bs];
  }

  printf("two_p n=%zu min %f max %f\n", b->size(), min, max);
}

void glImage::rebuffer(){
  myBi = parentZprInstance->myBi;
  int NRow = image->NRow; int NCol = image->NCol;
  printf("=======================================\n");
  printf("glImage::rebuffer() %d %d %d nr %d nc %d\n", myBi->at(0), myBi->at(1), myBi->at(2), NRow, NCol);
  SA< SA<float> * > * FB = image->getFloatBuffers();
  SA<float> * b1 = FB->at(myBi->at(0));
  SA<float> * b2 = FB->at(myBi->at(1));
  SA<float> * b3 = FB->at(myBi->at(2));
  float max1, max2, max3, min1, min2, min3, r1, r2, r3;
  r1 = r2 = r3 = 1.;
  min1 = min2 = min3 = 0.;

  int is_scene = strncmp(parentZprInstance->getTitle().c_str(), "Scene", 5) == 0;
  if(USE_PROPORTIONAL_SCALING){
    if(is_scene){
      two_percent(min1, max1, b1, b2, b3);
      parentZprInstance->image_intensity_min = min1;
      parentZprInstance->image_intensity_max = max1;
      printf("\t\tmin %f max %f\n", min1, max1);

    }
    else{
      zprInstance * scene = myZprManager->at(0);
      int scene_is_scene = strncmp(scene->getTitle().c_str(), "Scene", 5) == 0;
      if(!scene_is_scene){
        cout << "WARNING1: titles not initialized yet" << endl;
        return; // get out if titles not initialized yet
      }
      min1 = scene->image_intensity_min;
      max1 = scene->image_intensity_max;
    }
    min2 = min3 = min1; // proportional scaling: same for all bands
    max2 = max3 = max1;
  }
  else{

    if(is_scene){
      two_percent(min1, max1, b1); // so the 2p stretch happens in the secondary buffer (this one)
      two_percent(min2, max2, b2);
      two_percent(min3, max3, b3);
      zprInstance * p = parentZprInstance;
      p->image_intensity_min1 = min1;
      p->image_intensity_min2 = min2;
      p->image_intensity_min3 = min3;
      p->image_intensity_max1 = max1;
      p->image_intensity_max2 = max2;
      p->image_intensity_max3 = max3;
    }
    else{
      zprInstance * s = myZprManager->at(0);
      int s_is_scene = strncmp(s->getTitle().c_str(), "Scene", 5) == 0;
      if(!s_is_scene){
        cout << "WARNING: TITLES NOT INITIALIZED YET" << endl;
        return; // get out if titles not initialized yet
      }
      min1 = s->image_intensity_min1;
      min2 = s->image_intensity_min2;
      min3 = s->image_intensity_min3;
      max1 = s->image_intensity_max1;
      max2 = s->image_intensity_max2;
      max3 = s->image_intensity_max3;
    }

  }

  r1 = 1./(max1 - min1);
  r2 = 1./(max2 - min2);
  r3 = 1./(max3 - min3);

  printf("myZprInstanceID %d r1 %f r2 %f r3 %f min1 %f min2 %f min3 %f\n", parentZprInstance->myZprInstanceID, r1, r2, r3, min1, min2, min3);

  float r, g, b;
  long int i, j, k, ri, m;
  int class_i, n_class, gi;
  k = m = 0;

  for(i = 0; i < NRow; i++){
    ri = NRow - i - 1;
    for(j = 0; j < NCol; j++){
      r = r1 * (b1->at(ri, j) - min1);
      r = (r < 0. ? 0.: r);
      r = (r > 1. ? 1.: r);
      dat->at(k++) = r;

      g = r2 * (b2->at(ri, j) - min2);
      g = (g < 0. ? 0.: g);
      g = (g > 1. ? 1.: g);
      dat->at(k++) = g;

      b = r3 * (b3->at(ri, j) - min3);
      b = (b < 0. ? 0.: b);
      b = (b > 1. ? 1.: b);
      dat->at(k++) = b;

      // assign a groundref label to each pixel, if the pixel isn't confused
      // 0. for no affirmative label, -1 for multiple affirmative labels

      class_i = -1; // -1: affirmative gt value not yet found
      n_class = 0; // number of gt-affirmative matches

      for(gi = 0; gi < groundref.size(); gi++){
        if(groundref_disable.count(gi)) continue;
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
  // printf("\tdone rebuffer\n");
}

zprInstance * zprManager::newZprInstance(int NROW, int NCOL, int NBAND){
  int myWindowWidth = NCOL;
  int myWindowHeight = NROW;
  int myDims = NBAND;

  // dprintf("zprManager::newZprInstance(%d,%d)", myWindowWidth, myWindowHeight);

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

  // dprintf("zprManager::newZprInstance: newWindowID (%d), myZprInstanceID (%d), nextZprInstanceID (%d), myZprWindowID_AsFunctionOfGlutID[ newWindowID](%d), myGlutID_AsFunctionOfZprWindowID[nextZprInstanceID](%d)", newWindowID, ret->myZprInstanceID, nextZprInstanceID, myZprWindowID_AsFunctionOfGlutID[newWindowID], myGlutID_AsFunctionOfZprWindowID[ret->myZprInstanceID]);
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
    if((*it)->forceUpdate){
      forceUpdate = true;
      myZprManager->forceUpdate = true;
    }
    i++;
  }
}

void zprInstance::drawGraphicsExternal(){
  int zprID = myZprManager->getActiveZprInstanceID();
  int activeGLUTID = glutGetWindow();
  if(myGlutID() == activeGLUTID) return;
  glutSetWindow(myGlutID());
  // dprintf("Draw graphics external.. Caller ID (%d) My ID (%d)", activeGLUTID, myGlutID());

  GLERROR;
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  std::vector<glPlottable *>::iterator it;
  int i=0;
  mark();
  for(it = myGraphics.begin(); it!=myGraphics.end(); it++){
    ++i;
    if((*it)->isLinkd){
      // dprintf(" Drawing LINKED item (%d of %d) \n",i,myGraphics.size());
      (*it)->drawMe(true);
    }
    else{
      (*it)->drawMe();
    }
  }
  glutSwapBuffers();
  GLERROR;
  glutSetWindow(activeGLUTID);
  // dprintf("Return from drawGraphics()");
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
  this->drawGraphics();
  this->drawText();
  glutSwapBuffers();
  GLERROR;
  if(!forceUpdate){
    refreshflag = false;
  }
}

void zprInstance::idle(){
  if(refreshflag){
    glFlush();
    glutPostRedisplay();
  }
}

void zprInstance::setrgb(int r, int g, int b){

  for(int i=0; i < myZprManager->nextZprInstanceID; i++){
    zprInstance * a = myZprManager->at(i);
    cout << getTitle() << "(" << myGlutID() <<")" << "::setrgb()\n";
    a->myBi->at(0) = r;
    a->myBi->at(1) = g;
    a->myBi->at(2) = b;

    string s(a->getTitle().substr(0, 6)); // update display title
    str rs(vec_band_names[r]);
    str gs(vec_band_names[g]);
    str bs(vec_band_names[b]);

    a->setTitle(s + str("R,G,B=[")
    + to_string(r + 1) + str(":") + rs.substr(0, 31) + str(", ")
    + to_string(g + 1) + str(":") + gs.substr(0, 31) + str(", ")
    + to_string(b + 1) + str(":") + bs.substr(0, 31) + str("]"));

    // trickle-down. N.b. the glImage()::rebuffer() gets band-select info from zprInstance
    for(vector<glPlottable *>::iterator it = a->myGraphics.begin(); it != a->myGraphics.end(); it++){
      if((*it)->myType.compare(std::string("glImage")) == 0){
        ((glImage *)((void *)((glPlottable *)(*it))))->rebuffer();
      }
    }
  }
  for(int m = 0; m < 5; m++){
    zprInstance * a = myZprManager->myZprInstances->at(m);
    if(a != this){
      a->focus();
      a->mark();
      a->display();
    }
  }

  this->focus();
  this->mark();
  this->display();
}

void zprInstance::getrgb(int & r, int & g, int & b){
  r = myBi->at(0);
  g = myBi->at(1);
  b = myBi->at(2);
}

void zprInstance::processString(){
  int i = 0;
  char strSleep[] = "sleep";
  if(console_string[i] == '\0'){
    isPaused = !isPaused;
    printf("isPaused=%d\n", (int)isPaused);
    return;
  }
  while(console_string[i] != '\0') i++;
  int count = i + 1;

  SA<char> s(count);
  strcpy(&s[0], console_string);

  int r, g, b, tk;
  int ndim = NBand;

  // does string have "sleep" prefix?
  if(strcmpz(&s[0], &strSleep[0])){
    int st = atoi(&s[5]);
    if(st < 1) return;
    usleepNupdate = st;
    return;
  }

  // i prefix?
  if(strcmpz(console_string, "i\0")){
    // iterate analysis window!
    printf("iterate analysis window..\n");

    int half_win = (NWIN - 1) / 2;
    long int ii, jj;
    int n_inc = half_win;

    for(ii =0; ii < SUB_MM; ii += n_inc){
      if(ii - half_win >= 0 && ii - half_win - 1 + NWIN < SUB_MM){
        for(jj = 0; jj < SUB_MM; jj += n_inc){
          if(jj - half_win >= 0 && jj - half_win - 1 + NWIN < SUB_MM){

            printf("i %ld j %ld\n", ii, jj);
            WIN_I = ii - half_win;
            WIN_J = jj - half_win;

            if(true){
              SA<float> * dat4 = TGT;
              size_t i, j, k;

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

              for(int k = 0; k <= 4; k++){
                zprInstance * a = myZprManager->myZprInstances->at(k);
                a->focus();
                a->mark();
                a->display();
              }

              focus();
              mark();
              display();
              myZprDisplay();

            }

          }
        }
      }
    }
    // WIN_I = (y + NWIN) >= SUB_MM ? SUB_MM - NWIN : y;
    // WIN_J = (x + NWIN) >= SUB_MM ? SUB_MM - NWIN : x;
  }

  // s prefix?
  if(strcmpz(console_string, "s\0") && console_string[2] == '\0'){
    USE_PROPORTIONAL_SCALING = !USE_PROPORTIONAL_SCALING;
    printf("PROPORTIONAL_SCALING: %s\n", USE_PROPORTIONAL_SCALING?"On":"Off");
    int r, g, b;
    getrgb(r,g,b);
    setrgb(r,g,b);
    return;
    /*

    // trickle-down. N.b. the glImage()::rebuffer() gets band-select info from zprInstance
    for(vector<glPlottable *>::iterator it = myGraphics.begin(); it != myGraphics.end(); it++){
      if((*it)->myType.compare(std::string("glImage")) == 0){
        ((glImage *)((void *)((glPlottable *)(*it))))->rebuffer();
      }
    }

    for(int m = 0; m < 5; m++){
      zprInstance * a = myZprManager->myZprInstances->at(m);
      if(a != this){

        a->focus();
        a->mark();
        a->display();
      }
    }

    this->focus();
    this->mark();
    this->display();
    */
  }

  // gt prefix?
  if(strcmpz(console_string, "gt\0")){
    if(console_string[2] == '\0'){
      // toggle gt colouring
      groundref_class_colouring = !groundref_class_colouring;
      if(!groundref_class_colouring){
        groundref_disable.clear(); // enable all groundref bands, if colouring turned off
      }

    }
    else{
      int gi = -1; // see if the argument matches one of the band names
      for(int k = 0; k < groundref.size(); k++){
        // ground ref band name
        const char * grbn = vec_band_names[groundref[k]].c_str();
        if(strcmpz(&console_string[2], grbn)){
          gi = k;
          break;
        }
        printf("%d%s\n", k, grbn);
      }

      // printf("%s\n", &console_string[2]);
      if(gi >= 0){
        // had a match on one of the groundref names
        if(groundref_disable.count(gi)){
          groundref_disable.erase(gi);
        }
        else{
          groundref_disable.insert(gi);
        }
        for(set<int>::iterator it = groundref_disable.begin(); it != groundref_disable.end(); it++){
          cout << "***" << *it << endl;
        }
      }
      else{
        // didn't match on groundref name. Is this a number?
        str y(&console_string[2]);
        if(is_int(y)){
          int gi = atoi(y.c_str());
          if(gi >=0 && gi < groundref.size()){
            if(groundref_disable.count(gi)){
              groundref_disable.erase(gi);
            }
            else{
              groundref_disable.insert(gi);
            }
            for(set<int>::iterator it = groundref_disable.begin(); it != groundref_disable.end(); it++){
              cout << "***" << *it << endl;
            }
          }
        }
      }

    }
    // rebuffer the related glImage here!
    std::vector<glPlottable *>::iterator it;
    for(it = myGraphics.begin(); it!=myGraphics.end(); it++){
      if((*it)->myType == string("glPoints")){
        ((glPoints *)(*it))->myI->rebuffer();
      }
    }
    // done rebuffer
  }

  i = atoi(&s[1]); // see if string is of form xyy..yy where x is char and yy.yy contains int
  switch(s[0]){
    case 'd': resetViewPoint();
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
    if((i < 1) || (i > ndim)) break;
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
    // console input is just a number: set r, g, b to same band index
    try{
      i = atoi(&s[0]);
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
  int r, g, b;
  if(key == GLUT_KEY_F1){
    _F1 = true;
    printf("Move atom active\n");
    //printf("Special Key dn: %d, %d\n", key, _F1);
  }
  if(key == GLUT_KEY_F2){
    if(!_F2){
      _F2 = true; // !_F2;
    }
  }

  switch(key){
    case GLUT_KEY_UP: // "next frame" if applicable
    if(true){
      getrgb(r, g, b);
      r += bands_per_frame; // increment
      g += bands_per_frame;
      b += bands_per_frame;
      if(r >= NBand) r -= NBand; // check if wrap
      if(g >= NBand) g -= NBand;
      if(b >= NBand) b -= NBand;
      setrgb(r, g, b);
      printf("incremented band selection: (r,g,b)=(%d,%d,%d)\n", r, g, b);
    }
    break;

    case GLUT_KEY_DOWN: // "last frame" if applicable
    if(true){
      getrgb(r, g, b);
      r -= bands_per_frame; // decrement
      g -= bands_per_frame;
      b -= bands_per_frame;
      if(r < 0) r += NBand; // check if wrap
      if(g < 0) g += NBand;
      if(b < 0) b += NBand;
      setrgb(r, g, b);
      printf("decremented band selection: (r,g,b)=(%d,%d,%d)\n", r, g, b);
    }
    break;

    case GLUT_KEY_LEFT: // decrement selected-band indices
    if(true){
      getrgb(r, g, b);
      r--; // decrement
      g--;
      b--;
      if(r < 0) r += NBand; // check if wrap
      if(g < 0) g += NBand;
      if(b < 0) b += NBand;
      setrgb(r, g, b);
      printf("decremented band selection: (r,g,b)=(%d,%d,%d)\n", r, g, b);
    }
    break;

    case GLUT_KEY_RIGHT: // increment selected-band indices
    if(true){
      getrgb(r, g, b);
      r++; // increment
      g++;
      b++;
      if(r >= NBand) r -= NBand; // check if wrap
      if(g >= NBand) g -= NBand;
      if(b >= NBand) b -= NBand;
      setrgb(r, g, b);
      printf("incremented band selection: (r,g,b)=(%d,%d,%d)\n", r, g, b);
    }
    break;

    default: break;
  }
}

void zprInstance::specialUp(int key, int x, int y){
  refreshflag = true;
  if( key ==GLUT_KEY_F1){
    _F1 = false;
  }

  if(key == GLUT_KEY_F2){
    _F2 = false;
  }

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
    console_position++;
    console_string[console_position] = '\0';

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
    exit(0); //printf( "%d Pressed Esc\n",(char)key);
    break;

    // Delete
    /* case 127 :
    printf( "%d Pressed Del\n",(char)key);
    break; */
    default: //printf( "Pressed key %c AKA %d at position %d % d\n",(char)key, key, x, y);
    console_string[console_position++] = (char)key;
    console_string[console_position]='\0'; // printf("STRING: %s\n", &console_string[0]);
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
void zprInstance::setTitle(string s){
  focus();
  myTitle = s;
  glutSetWindowTitle(s.c_str());
}

void zprInstance::zprReshape(int w,int h){
  // focus();
  // printf("zprReshape\n");
  GLfloat ratio; // http://faculty.ycp.edu/~dbabcock/cs370/labs/lab07.html
  //glPushMatrix();
  glViewport(0,0,w,h); // Set new screen extents

  // Select projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //getMatrix(); // added 20200619 from cymh_cli_utils

  // Adjust viewing volume (orthographic)
  // If taller than wide adjust y
  if(w <= h){
    ratio = (GLfloat) h / (GLfloat) w;
    glOrtho(-1.0f, 1.0f, -1.0f * ratio, 1.0f * ratio, -1.0f, 1.0f);
    _bottom = -1. * ratio; _top = 1. * ratio;
    _left = -1.; _right = 1.;
  }
  // If wider than tall adjust x
  else if (h <= w){
    ratio = (GLfloat) w / (GLfloat) h;
    glOrtho(-1.0f * ratio, 1.0f * ratio, -1.0f, 1.0f, -1.0f, 1.0f);
    _left = -1. * ratio; _right = 1. * ratio;
    _bottom = -1.; _top = 1.;
  }

  NRow = h;
  NCol = w; // might need to disable this for specific windows?

  glMatrixMode(GL_MODELVIEW); // put back in 20200619
  refreshflag = true;
  return;

}
// http://graphics.stanford.edu/courses/cs248-01/OpenGLHelpSession/code_example.html

void zprInstance::zprMouse(int button, int state, int x, int y){
  // printf("zprMouse()\n");
  GLint viewport[4]; /* Do picking */
  refreshflag = true;

  int is_analysis = strncmp(getTitle().c_str(), "Analys", 6) == 0;
  if(is_analysis) return;

  if(state == GLUT_DOWN && button == GLUT_LEFT_BUTTON){
    zprPick(x, glutGet(GLUT_WINDOW_HEIGHT) - 1 - y, 3, 3);
    _pickme(0);
  }
  else{
    pick(-1);
  }

  _mouseX = x;
  _mouseY = y;

  if(state == GLUT_UP){
    switch(button){
      case GLUT_LEFT_BUTTON: _mouseLeft = false; myPickNames.clear(); break;
      case GLUT_MIDDLE_BUTTON: _mouseMiddle = false; break;
      case GLUT_RIGHT_BUTTON: _mouseRight = false; break;
    }
  }
  else{
    switch(button){
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

    // now rebuffer under secondary window too
    WIN_I = (y + NWIN) >= SUB_MM ? SUB_MM - NWIN : y;
    WIN_J = (x + NWIN) >= SUB_MM ? SUB_MM - NWIN : x;

    size_t i, j, k;
    SA<float> * dat4 = TGT;
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

    for(int m = 0; m <= 4; m++){
      // if(m > 1) continue; // update the first two windows (otherwise get segfault)
      zprInstance * a = myZprManager->myZprInstances->at(m);
      a->focus();
      a->mark();
      a->display();
    }
    //myZprDisplay();
    this->focus();
    this->mark();
    this->display();

  }

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

    for(int k = 0; k <= 4; k++){
      zprInstance * a = myZprManager->myZprInstances->at(k);
      a->focus();
      a->mark();
      a->display();
    }

    focus();
    mark();
    display();
    myZprDisplay();

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
      double d = (double)(FB->at(i)->at((y * NCol) + x));
      printf("%d\t%s\t%e\n", i, vec_band_names[i].c_str(), d);
    }
  }

  // update windows
  /*
  for(int m = 2; m < myZprManager->myZprInstances->size(); m++){

    zprInstance * s = myZprManager->myZprInstances->at(3);
    if(s->myZprInstanceID == myZprInstanceID) continue;
    s->focus();
    s->mark();
    s->display();
  }

  focus();
  mark();
  display();
  */

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
        changed = true;
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
        changed = true;
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
  for(c = string; *c != '\0'; c++){
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
  // printf("glPoints::drawMe()\n");
  size_t nr = myI->image->NRow;
  size_t nc = myI->image->NCol;
  size_t nf = nr * nc * 3;
  float * d = myI->dat->elements;
  float r, g, b, h, s, v;
  float class_label;

  float m;
  glColor3f(1,1,1);
  glPointSize(2.);
  // if(!myI->myParent->groundref_class_colouring)
  if(!parentZprInstance->groundref_class_colouring){
    // plot data points with color: (r,g,b) = (x,y,z)
    for(size_t i = 0; i < nf; i += 3){
      r = d[i];
      g = d[i + 1];
      b = d[i + 2];
      m = max3(r, g, b);
      glColor3f(r/m, g/m, b/m); // g, b);
      glBegin(GL_POINTS);
      glVertex3f(r, g, b);
      glEnd(); // printf("%f %f %f]\n", d[i], d[i + 1], d[i + 2]);
    }
  }
  else{
    // plot data points colored by GT label
    //
    // todo: translate disabled list, into "enabled" list, and recolour based on that!
    s = v = 1.;
    d = myI->class_label->elements;
    float * dd = myI->dat->elements;

    float ci = 0.;
    int n_groundref_enabled = groundref.size() - groundref_disable.size();
    map<float, float> class_label_mapping; // for(size_t i = 0; i < groundref.size(); i++) if(groundref_disable.count(i))
    class_label_mapping.clear();
    for(int i = 0; i < groundref.size(); i++){
      if(groundref_disable.count(i)){
      }
      else{
        if(class_label_mapping.count(i) < 1){
          class_label_mapping[i] = ci ++;
        }
      }
    }
    map<float, float> groundref_index_map;
    for(int i = 0; i < groundref.size(); i++){
      groundref_index_map[groundref[i]] = i;
    }

    cout << "class_label_mapping: " << class_label_mapping << endl;
    cout << "groundref_disable: " << groundref_disable << endl;
    cout << "groundref_index_map: " << groundref_index_map << endl;

    map<float, float> labels_in_window;
    for(size_t i = 0; i < nf; i += 3){
      class_label = d[i / 3];
      if(labels_in_window.count(class_label) < 1){
        labels_in_window[class_label] = 0.;
      }
      labels_in_window[class_label] +=1.;
    }
    cout << "labels_in_window: " << labels_in_window << endl;
    for(size_t i = 0; i < nf; i += 3){
      class_label = d[i / 3];
      if(groundref_disable.count(class_label -1)) continue; // skip plot if this class disabled
      if(class_label == 0.){
        r = g = b = 0.; // what does this code mean?
      }
      else if(class_label == -1.){
        r = g = b = 1.; // is -1 class label N/A?
      }
      else{
        // h = 360. * (float)(class_label - 1) / (float)(groundref.size());
        h = 360. * (float)(class_label_mapping[class_label - 1]) / (float)(n_groundref_enabled);
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

    // colored legend!!!!!!
    for(size_t i = 0; i < groundref.size(); i++){
      if(groundref_disable.count(i)) continue; // skip this one if we indicate to skip it!

      // however, should also skip this class in selecting the label for this pix. Did we?
      // that would be in the refreshing of the glPoints data
      int ii = (int)i;
      str pre(str("(") + to_string((int)i) + str(") "));
      str class_str(pre + vec_band_names[groundref[i]]);
      const char * class_string = class_str.c_str();
      // h = 360. * (float)(i) / (float)(groundref.size());
      h = 360. * (float)(class_label_mapping[i]) / (float)(n_groundref_enabled);
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
