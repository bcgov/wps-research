/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library
that inspired further developments at UVic, CFS and elsewhere.. */
#pragma once

#ifndef NEWZPR_H
#define NEWZPR_H

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>
#include "my_math.h"
#include "SA.h"
#include "aestheticparameters.h"
#include "pthread.h"
#include "time.h"
#include "image.h"
#include<queue>
#include"util.h"

using namespace std;

void GLERROR();

#define MAXBUFFERSIZE 1000000
#define MAX_ZPR_INSTANCES 100
//#define STARTX 400
//#define STARTY 400
#define STR_MAX 1000
//#define MYFONT GLUT_BITMAP_HELVETICA_18
//#define MYFONT GLUT_BITMAP_8_BY_13
//#define MYFONT GLUT_BITMAP_9_BY_15
//#define MYFONT GLUT_BITMAP_TIMES_ROMAN_10
//#define MYFONT GLUT_BITMAP_TIMES_ROMAN_24
//#define MYFONT GLUT_BITMAP_HELVETICA_10
#define MYFONT GLUT_BITMAP_HELVETICA_12
//define MYFONT GLUT_BITMAP_HELVETICA_18
///ii=frameIndex.begin();/ii=frameIndex.begin();
static clock_t start_time;
static clock_t stop_time;
#define SECONDS_PAUSE 0.4

// image subset selection parameters
extern size_t SUB_START_I;
extern size_t SUB_START_J;
extern float SUB_SCALE_F;
extern size_t SUB_MM;
extern size_t IMG_NR;
extern size_t IMG_NC;
extern size_t IMG_NB;
extern SA<float> * IMG;
extern SA<float> * SUB;
extern myImg * SUB_MYIMG;
extern void * SUB_GLIMG;
extern string IMG_FN;

// target window parameters
extern size_t NWIN;
extern size_t WIN_I;
extern size_t WIN_J;
extern SA<float> * TGT;
extern myImg * TGT_MYIMG;
extern void * TGT_GLIMG;

// groundref detection
extern vector<int> groundref;

// stuff
class zprManager;
extern zprManager * myZprManager; // = NULL;
class zprInstance;

static int fullscreen = 0;

static void myZprReshape(int w,int h);//
static void myZprMouse(int button, int state, int x, int y);
static void myZprMotion(int x, int y);
static void my_pick(GLint name);

static GLfloat light_ambient[] = {
  0.0,
  0.0,
  0.0,
  1.0
};

static GLfloat light_diffuse[] = {
  1.0,
  1.0,
  1.0,
  1.0
};

static GLfloat light_specular[] = {
  1.0,
  1.0,
  1.0,
  1.0
};

static GLfloat light_position[] = {
  1.0,
  1.0,
  1.0,
  0.0
};

static GLfloat mat_ambient[] = {
  0.7,
  0.7,
  0.7,
  1.0
};

static GLfloat mat_diffuse[] = {
  0.8,
  0.8,
  0.8,
  1.0
};

static GLfloat mat_specular[] = {
  1.0,
  1.0,
  1.0,
  1.0
};

static GLfloat high_shininess[] = {
  100.0
};

static int strcmpz(const char * a, const char * b){
  int i=0;
  while(a[i]!=NULL && b[i] !=NULL && a[i]==b[i]){
    i++;
  }
  if(b[i]==NULL){
    return(true);
  }
  else{
    return(false);
  }
}

static void initLighting(){
  GLERROR();

  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);

  glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
  glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
  glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glDepthFunc(GL_LESS);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glEnable(GL_COLOR_MATERIAL);

  GLERROR();
}

static void quitme(){
}

class zprManager{

  std::map<int, int> myZprWindowID_AsFunctionOfGlutID;
  std::map<int, int> myGlutID_AsFunctionOfZprWindowID;
  std::vector<int> myGlutWindowIDsInOrder;

  public:
  int nextGLUTName;
  int forceUpdate;

  std::set<int> myGlutWindowIDs;
  SA< zprInstance * > * myZprInstances;

  int getActiveZprInstanceID(){
    int currentID = glutGetWindow();
    int zprInstanceID = myZprWindowID_AsFunctionOfGlutID[currentID];
    if(myGlutID_AsFunctionOfZprWindowID[zprInstanceID]!=currentID){
      dprintf("Error: glutWindowID (%d) vs. myGlutID_AsFunctionOfZprWindowID[zprInstanceID] (%d) mismatch.\nmyZprWindowID_AsFunctionOfGlutID[currentID]=%d", currentID, myGlutID_AsFunctionOfZprWindowID[zprInstanceID],myZprWindowID_AsFunctionOfGlutID[currentID]);
      exit(1);
    }
    return(zprInstanceID);
  }

  void zprReshape(int w, int h);
  void zprMouse(int button, int state, int x, int y);
  void zprMotion(int x, int y);
  void _pick(GLint name);
  void zprKeyboard(unsigned char key, int x, int y);
  void zprKeyboardUp(unsigned char key, int x, int y);
  void zprSpecial(int key, int x, int y);
  void zprSpecialUp(int key, int x, int y);
  void zprDisplay();
  void zprIdle();
  void zprDrawGraphics();

  private:
  int nextZprInstanceID;

  zprManager(int argc, char *argv[]){
    forceUpdate = false;
    cout <<"zprManager()\n";
    nextGLUTName = 0;
    myZprWindowID_AsFunctionOfGlutID.clear();
    myGlutID_AsFunctionOfZprWindowID.clear();
    myGlutWindowIDs.clear();
    myZprInstances = new SA< zprInstance * >((int)MAX_ZPR_INSTANCES);
    int i;
    for(i = 0; i < MAX_ZPR_INSTANCES; i++){
      (*myZprInstances)[i] = NULL;
    }
    nextZprInstanceID = 0;
    printf("glutInit()\n");
    glutInit(&argc, argv);
    printf("after glutInit()\n");
  }

  public:
  void focus(int myID){
    glutSetWindow(myID);
  }

  public:

  void mark();
  static zprManager * Instance(int argc, char *argv[]);
  zprInstance * newZprInstance();
  zprInstance * newZprInstance(int NROW, int NCOL, int NBAND);
};

static void myZprReshape(int w,int h){
  myZprManager->zprReshape(w,h);
}

static void myZprMouse(int button, int state, int x, int y){
  myZprManager->zprMouse(button, state, x, y);
}

static void myZprMotion(int x, int y){
  myZprManager->zprMotion(x,y);
}

static void my_pick(GLint name){
  myZprManager->_pick(name);
}

static void myZprKeyboard(unsigned char key, int x, int y){
  myZprManager->zprKeyboard(key, x,y);
}

static void myZprKeyboardUp(unsigned char key, int x, int y){
  myZprManager->zprKeyboardUp(key, x,y);
}

static void myZprSpecial(int key, int x, int y){
  myZprManager->zprSpecial(key, x,y);
}

static void myZprSpecialUp(int key, int x, int y){
  myZprManager->zprSpecialUp(key, x,y);
}

static void myZprDisplay(){
  myZprManager->zprDisplay();
}

static void myZprIdle(){
  myZprManager->zprIdle();
}

static void myZprDrawGraphics(){
  myZprManager->zprDrawGraphics();
}

class glPlottable;

class knnGLUT2d;
class knnGLUT3d;
class knnClusteringInstance;

class zprInstance{
  public:
  void (*callMeWithMouseClickCoordinates)(int, int);
  float myLeft, myRight, myBottom, myTop, myZNear, myZFar;
  int usleepN, usleepNupdate;
  int isPaused;
  int forceUpdate;
  SA<int> * myBi;
  int ZPR_ZOOM_MODE, ZPR_PAN_MODE, ZPR_ROTATE_MODE;
  int NBand, NRow, NCol;
  set<int> * myLabelsPicked;
  knnClusteringInstance * myKnnClusteringInstance;

  int getPickedLabel(){
    if(myLabelsPicked->size() > 0){
      std::set<int>::iterator ti;
      ti = myLabelsPicked->begin();
      return *ti;
    }
    else return -1;
  }

  knnGLUT2d * myglut2d;
  knnGLUT3d * myglut3d;
  zprManager * myZprManager;
  vector <glPlottable *> myGraphics;
  std::map<GLint, glPlottable* > myGraphicsAsAFunctionOfGLName;
  vector <glPlottable *> myGraphicsLabelled;
  int refreshflag;
  char console_string[STR_MAX];
  int console_position;
  int myScreenPositionX, myScreenPositionY;
  int myZprInstanceID;
  int myGLUTWindowID;
  std::set<GLint> myPickNames;
  vec3 rX;
  int myWindowWidth; int myWindowHeight;
  int _F1; //F1 key active?
  int _mouseX,_mouseY, _mouseLeft, _mouseMiddle, _mouseRight;
  double _left,_right, _top,_bottom,_near,_far,_zNear,_zFar, _dragPosX, _dragPosY, _dragPosZ;
  double _matrix[16];
  double _matrixInverse[16];
  void zprInit();
  GLfloat zprReferencePoint[4];

  /* Picking API (Optional) */
  void zprSelectionFunc(void (*f)(void));
  void zprPickFunc(void (*f)(GLint name));
  void (*pick)(GLint name);
  void (*selection)(void);

  int nextGLUTName(){
    return(myZprManager->nextGLUTName);
  }

  void advanceGLUTName(){
    myZprManager->nextGLUTName++;
  }

  void add( glPlottable * a);

  void setTitle(string t); // set title string

  double vlen(double x,double y,double z);
  void pos(double *px,double *py,double *pz,const int x,const int y,const int *viewport);
  void getMatrix();
  void invertMatrix(const GLdouble *m, GLdouble *out );
  void zprReshape(int w,int h);
  void zprMouse(int button, int state, int x, int y);
  void zprMotion(int x, int y);
  void _pickme(GLint name);
  void zprPick(GLdouble x, GLdouble y,GLdouble delX, GLdouble delY);
  void processHits(GLint hits, GLuint buffer[]);

  void zprInstanceInit(int ZprID, int glutID, zprManager * manager, int _NROW, int _NCOL, int nb){
    printf("zprInstanceInit ZprID %d glutID %d\n", ZprID, glutID);
    callMeWithMouseClickCoordinates = NULL;
    myLabelsPicked = new set<int>;
    myLabelsPicked->clear();
    myPickNames.clear();
    myKnnClusteringInstance = NULL;
    myBi = new SA<int>(3);
    (*myBi)[0] = 0;
    (*myBi)[1] = 1;
    (*myBi)[2] = 2;
    selection = NULL;
    myglut2d = NULL;
    myglut3d = NULL;
    myGraphics.clear();
    console_position = 0;
    console_string[0]='\0';
    myZprInstanceID = ZprID;
    myGLUTWindowID = glutID;
    myZprManager = manager;
    myWindowHeight = _NROW;
    myWindowWidth = _NCOL;
    NRow = _NROW;
    NCol = _NCOL;
    NBand = nb;
    myScreenPositionX= 0;
    myScreenPositionY = 0;
    _F1 = false;
    _mouseX = 0;
    _mouseY = 0;
    _mouseLeft = 0;
    _mouseMiddle = 0;
    _mouseRight = 0;
    _dragPosX = 0.0;
    _dragPosY = 0.0;
    _dragPosZ = 0.0;
    rX.init(0., 0., 0.);
    refreshflag = true;
    ZPR_ZOOM_MODE = ZPR_PAN_MODE = ZPR_ROTATE_MODE = false;
    isPaused = false;
    forceUpdate = false;
    usleepN = 15000;
    usleepNupdate = usleepN;
    myLeft= FLT_MAX;
    myRight=FLT_MIN;
    myBottom=FLT_MAX;
    myTop=FLT_MIN;
    myZNear=0.;
    myZFar=1.;
  }

  zprInstance(int ZprID, int glutID, zprManager * manager, int _NROW, int _NCOL, int nb){
    zprInstanceInit(ZprID, glutID, manager, _NROW, _NCOL, nb);
  }

  void setTitle( char * s){
    char asdf[1000]="\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    sprintf(&asdf[0], "GLUTID (%d) %s\0", myGlutID(), s);
    glutSetWindowTitle(&(asdf[0]));
  }

  void mark();

  void resetViewPoint(){
    refreshflag=true;
    mark();
    myWindowWidth = glutGet( GLUT_WINDOW_WIDTH );
    myWindowHeight = glutGet( GLUT_WINDOW_HEIGHT );
    GLdouble aspect = (float)myWindowWidth/myWindowHeight;
    vec3 c( (myRight+myLeft)/2., (myTop+myBottom)/2., (myZFar + myZNear)/2.);
    vec3 r( myRight-myLeft, myTop-myBottom, myZFar - myZNear);
    vec3 a( abs(r.x), abs(r.y), abs(r.z));
    float diam =2.* max(a.z,max(a.x,a.y));
    _left = c.x - diam;
    _right = c.y+diam;
    _bottom = c.y-diam;
    _top = c.y+diam;
    _zNear =2.*(diam+0.1);
    _zFar = -2.*(diam+0.1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if ( aspect < 1.0 ){
      // window taller than wide
      _bottom /= aspect;
      _top /= aspect;
    }
    else{
      _left *= aspect;
      _right *= aspect;
    }
    _zNear = 1.;
    _zFar = 0.;
    glOrtho(_left, _right, _bottom, _top, _zNear, _zFar);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    _mouseX = 0;
    _mouseY = 0;
    _mouseLeft = 0;//false;
    _mouseMiddle = 0;//false;
    _mouseRight = 0;//false;
    _dragPosX = 0.0;
    _dragPosY = 0.0;
    _dragPosZ = 0.0;
    rX.init(0., 0., 0.);
    getMatrix();
    refreshflag = true;
    mark();
    display();
    refreshflag = true;
    mark();
    return;
  }

  void resetView(){
    myWindowHeight = NRow; myWindowWidth = NCol;
    _mouseX = 0;
    _mouseY = 0;
    _mouseLeft = 0;//false;
    _mouseMiddle = 0;//false;
    _mouseRight = 0;//false;
    _dragPosX = 0.0;
    _dragPosY = 0.0;
    _dragPosZ = 0.0;
    rX.init(0., 0., 0.);//rX = rY = rZ =0.;
    refreshflag = true;
    this->zprInit();
  }

  void pilotView(GLdouble planex, GLdouble planey, GLdouble planez, GLdouble roll, GLdouble pitch, GLdouble heading){
    glRotated(roll, 0.0, 0.0, 1.0);
    glRotated(pitch, 0.0, 1.0, 0.0);
    glRotated(heading, 1.0, 0.0, 0.0);
    glTranslated(-planex, -planey, -planez);
  }
  //from openGL red book

  void polarView(GLdouble distance, GLdouble twist, GLdouble elevation, GLdouble azimuth){
    glTranslated(0.0, 0.0, -distance);
    glRotated(-twist, 0.0, 0.0, 1.0);
    glRotated(-elevation, 1.0, 0.0, 0.0);
    glRotated(azimuth, 0.0, 0.0, 1.0);
  }
  //from openGL red book
  //the zprInstance hits commands on the zprAble.

  int myZprId(){
    int zprID = myZprManager->getActiveZprInstanceID();
    if(zprID != myZprInstanceID){
      cout << "Error: local ID does not match active ID.\n"; exit(1);
    }
    return(zprID);
  }

  inline int myGlutID(){
    return myGLUTWindowID;
  }

  void focus(){
    glutSetWindow(myGlutID());
  }

  void glutGetWindowPosition(int &x, int &y);
  int setScreenPositionAndReturnBorderHeight(int x, int y);
  void setScreenPosition(int x, int y);
  void setRightOf( zprInstance * other);
  void setBelow( zprInstance * other);

  void drawGraphics(); void drawGraphicsExternal();
  void drawText();
  void display(void);
  void idle();

  void renderBitmapString(float x, float y, void *font, char *string);
  void setOrthographicProjection();
  void resetPerspectiveProjection();
  void processString();
  void keyboard(unsigned char key, int x, int y);
  void keyboardUp(unsigned char key, int x, int y);
  void special(int key, int x, int y);
  void specialUp(int key, int x, int y);
  int grabint(char * p );
  void setrgb(int r, int g, int b);
  void getrgb(int & r, int & g, int & b);

};

class glPlottable{
  public:

  vec3 x;
  int setRelativePosition;
  std::string myType;
  int hideMe; int forceUpdate;
  zprInstance * parentZprInstance;
  vec3 rgb;
  GLint myName;
  vector<glPlottable *> myLinks;
  vector<glPlottable *>::iterator linkIt;
  SA<int> * myBi;
  int isLabelled;
  int myLabel;
  int Update;
  int isLinkd; //flag to set coloring based on link-color activation.
  int isInverted;

  void mark(){
    Update = true; hideMe = false;
  }

  glPlottable(){
    myType = std::string("glPlottable");
    Update = false;
    isLabelled=false;
    isLinkd =false;
    myLinks.clear();
    hideMe = false;
    forceUpdate = false;
  }

  glPlottable(int isLbld){
    myType=std::string("glPlottable");
    glPlottable();
    isLabelled = isLbld;
    hideMe = false;
    forceUpdate = false;
  }

  void setLabel(int lab){
    myLabel = lab; isLabelled=true;
  }

  virtual void drawMe(){
  }

  virtual void drawMe(int highlight){
  }

  void setDefault(){
    hideMe = false;
    forceUpdate = false;
    isLabelled = false;
    Update = false;
    isLinkd = false;
    isInverted = false;
    myType=std::string("glPlottable");
  }

  void addPointToBoundingBox(float x, float y, float z){

    if(x<parentZprInstance->myLeft){
      parentZprInstance->myLeft=x;
    }
    if(x>parentZprInstance->myRight){
      parentZprInstance->myRight =x;
    }
    if(y<parentZprInstance->myBottom){
      parentZprInstance->myBottom = y;
    }
    if(y>parentZprInstance->myTop){
      parentZprInstance->myTop = y;
    }
    if(z<parentZprInstance->myZNear){
      parentZprInstance->myZNear = z;
    }
    if(z>parentZprInstance->myZFar){
      parentZprInstance->myZFar = z;
    }

    //printf("glortho(left=%f, right=%f, bottom=%f, top=%f, near=%f, far=%f\n",parentZprInstance->myLeft, parentZprInstance->myRight, parentZprInstance->myBottom, parentZprInstance->myTop, parentZprInstance->myZNear, parentZprInstance->myZFar);
  }

  void setRGB(float R, float G, float B){
    rgb.x=R;
    rgb.y=G;
    rgb.z=B;
  }

  void addLink( glPlottable * alink){
    myLinks.push_back(alink);
  }

  void linkMe(){
    if(isLinkd) return;
    isLinkd = true;
    if(myLinks.size()>0){
      for(linkIt=myLinks.begin(); linkIt!=myLinks.end(); linkIt++){
        (*linkIt)->linkMe();
        (*linkIt)->parentZprInstance->drawGraphicsExternal();
      }
    }
  }

  void unLink(){
    if(!isLinkd) return;
    isLinkd = false;
    if(myLinks.size()>0){
      for(linkIt=myLinks.begin(); linkIt!=myLinks.end(); linkIt++){
        (*linkIt)->unLink();
      }
    }
  }

  int isLinked(){
    return(isLinkd);
  }

  int isPicked(){
    int myCount = parentZprInstance->myPickNames.count(myName);
    if(myCount){
      if(isLabelled){
        parentZprInstance->myLabelsPicked->insert(myLabel);
        isLinkd = false;
        linkMe();
      }
    }
    else{
      if(isLabelled){
        parentZprInstance->myLabelsPicked->erase(myLabel);
        unLink();
      }
    }
    return(myCount);
  }

  void initName(zprInstance * parent, int useName);

  void colorMe(){
    if(isLinkd){
    }
    if( (this->isPicked()) || isLinkd || this->isInverted){
      glColor3f(1.-rgb.x,1.-rgb.y,1.-rgb.z);
    }
    else{
      glColor3f(rgb.x,rgb.y,rgb.z);
    }
  }

  void colorMe( int highlight){
    if( highlight || isInverted){
      glColor3f(1.-rgb.x,1.-rgb.y,1.-rgb.z);
    }
    else{
      glColor3f(rgb.x,rgb.y,rgb.z);
    }
  }

};

class myImg;

class glImage: public glPlottable{
  public:
  myImg * image;
  SA<float> * dat;
  int isClusteringImage;
  zprInstance *myParent;

  glImage(){
    dat = NULL;
    myType = std::string("glImage");
    Update = false; isClusteringImage = false;
  }

  glImage(zprInstance * parent, myImg * img ){
    myParent = parent;
    initFrom(parent,img);
    Update = false;
    isClusteringImage = false;
    myType = std::string("glImage");
  }

  void initFrom(zprInstance * parent, myImg * img){
    myParent = parent;
    image = img;
    dat = new SA<float>(image->NRow * image->NCol *image->NBand);
    initName(parent,false); Update = false; isClusteringImage = false;
    addPointToBoundingBox(0., 0., 0.);
    addPointToBoundingBox(0., (float)(image->NCol), 0.);
    addPointToBoundingBox((float)(image->NRow), (float)(image->NCol), 0.);
    addPointToBoundingBox((float)(image->NRow), 0., 0.);
    rebuffer();
  }

  void drawMeUnHide();

  void drawMe(){
    if(hideMe) return;
    if(Update) rebuffer();

    int NRow = image->NRow;
    int NCol = image->NCol;

    int nr = NRow; //myParent->NRow;
    int nc = NCol; //myParent->NCol;
    //printf("drawMe nrow %d ncol %d nr %d nc%d\n", NRow, NCol, nr, nc);

    glViewport(0, 0, nc, nr); //NCol, NRow);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // gluOrtho2D(0.0, (GLfloat)NCol, 0.0, (GLfloat)NRow);
    gluOrtho2D(0., (GLfloat)nc, 0., (GLfloat)nr);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRasterPos2f(0.,0.);
    glPixelStoref(GL_UNPACK_ALIGNMENT, 1);
    glDrawPixels(NCol, NRow, GL_RGB, GL_FLOAT, (GLvoid *)(&((dat->elements)[0])));

    /*
    extern int SUB_START_I;
    extern int SUB_START_J;
    extern float SUB_SCALE_F;
    extern int SUB_MM;
    */

    if(myParent->myZprInstanceID == 0){
      float x = SUB_SCALE_F * (float)SUB_START_J;
      float y = SUB_SCALE_F * (float)SUB_START_I;
      float w = SUB_SCALE_F * (float)SUB_MM;
      float h = SUB_SCALE_F *(float) SUB_MM;
      printf("x %f y %f w %f h %f\n", x, y, w, h);
      glLineWidth(1.5);
      glPushMatrix(); //Make sure our transformations don't affect any other transformations in other code
      glTranslatef(x, (float)NRow - y, 0);
      glColor3f(1., 0., 0.);
      //Put other transformations here
      glBegin(GL_LINES); //We want to draw a quad, i.e. shape with four sides
      glVertex2f(0, 0); //Draw the four corners of the rectangle
      glVertex2f(0, -h);
      glVertex2f(0, -h);
      glVertex2f(w, -h);
      glVertex2f(w, -h);
      glVertex2f(w, 0);
      glVertex2f(w, 0);
      glVertex2f(0, 0);
      glEnd();
      glPopMatrix();
    }

    // draw target/analysis window on subset window
    if(myParent->myZprInstanceID == 1){
      float x = (float)WIN_J;
      float y = (float)WIN_I;
      float w = (float)NWIN;
      float h = (float)NWIN;
      printf("target: x %f y %f w %f h %f NWIN %f\n", x, y, w, h, NWIN);
      glColor3f(1., 0., 0.);
      glLineWidth(1.5);
      glPushMatrix(); //Make sure our transformations don't affect any other transformations in other code
      glTranslatef(x, ((float)NRow) - y, 0);// (y + (h/2.)), 0);
      //Put other transformations here
      glBegin(GL_LINES); //We want to draw a quad, i.e. shape with four sides
      glVertex2f(0, 0); //Draw the four corners of the rectangle
      glVertex2f(0, -h);
      glVertex2f(0, -h);
      glVertex2f(w, -h);
      glVertex2f(w, -h);
      glVertex2f(w, 0);
      glVertex2f(w, 0);
      glVertex2f(0, 0);
      glEnd();
    }

  }

  void rebuffer();
};

/*
class glRect: public glPlottable{
  // two dimensional rectangle plotted in 2d only
  public:
  zprInstance *myParent;
  int x, y, w, h; // x position, y position, width, height

  glRect(){
    myType = std::string("glRect");
    Update = false;
  }

  glRect(zprInstance * parent, int x, int y, int w, int h){
    myParent = parent;
    myType = std::string("glRect");
  }

  void drawMeUnHide();

  void drawMe(){
    if(hideMe) return; // if(Update) rebuffer();

    int NRow = myParent->NRow;
    int NCol = myParent->NCol;

    cout << "glRect::drawMe()" << endl;
    glViewport(0,0,NCol, NRow);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLfloat)NCol, 0.0, (GLfloat)NRow);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 0, 1);
    glRectf((float)x, (float)y, (float)(x + w), (float)(y - h));
    // glRasterPos2f(0.,0.);
    //glPixelStoref(GL_UNPACK_ALIGNMENT, 1);
    //glDrawPixels(NCol, NRow, GL_RGB, GL_FLOAT, (GLvoid *)(&((dat->elements)[0])));

  }

  void rebuffer();
};

*/

class glSimpleRect: public glPlottable{
  // this is a rectangle plotted in up to 3d
  public:
  vec3 x;
  int myVal;
  int * myParam;
  int K1, K2;
  glSimpleRect(zprInstance * parent, float R, float G, float B, int myValue,int * _myParam, float a1, float a2, float a3, int k1, int k2){
    K1 = k1;
    K2 = k2;
    myParam = _myParam;
    setDefault();
    x.init(a1, a2, a3);
    myVal = myValue;
    rgb.init(R,G,B);
    parentZprInstance = parent;
    initName(parent,true);
    float j = (float)K1;
    float k = (float)K2;
    addPointToBoundingBox(a1, a2, a3+1.);
    addPointToBoundingBox(a1+j, a2+k, 0.);
  }

  void drawMeGL(){
    int isP = isPicked();

    if(isP){
      printf("glSimpleRect select (prev: %d) (new: %d) \n",*myParam, myVal);
      (*myParam) = myVal;
    }
    float j = (float)K1;
    float k = (float)K2;
    glBegin(GL_POLYGON);
    glVertex3f(x.x, x.y, x.z);
    glVertex3f(x.x+j, x.y, x.z);
    glVertex3f(x.x+j, x.y+k, x.z);
    glVertex3f(x.x , x.y+k, x.z);
    glEnd();
  }

  void drawMe(){
    colorMe();
    glPushMatrix();
    glPushName(myName);
    drawMeGL();
    glPopName();
    glPopMatrix();
  }
};

class glSlider{
  public:
  int nItems;
  int pixWidth; int blankWidth;
  int * myParam;
  SA< glSimpleRect *> * myRects;
  zprInstance * myZpr;

  glSlider(zprInstance * _myZpr, int nMax, int * paramVal, int pixW, int blankW){
    pixWidth = pixW; blankWidth = blankW;
    myParam = paramVal;
    myZpr = _myZpr;
    nItems = nMax;
    myRects = new SA<glSimpleRect * >(nMax);

    int i;
    for(i=0; i<nMax; i++){
      myRects->at(i)=NULL;
    }
    for(i=0; i<nItems; i++){
      vec3 rgb( 0., 1., 0.);//* (((float)i)+1.)/((float)nItems), 0.);
      (*myRects)[i] = new glSimpleRect( myZpr, rgb.x, rgb.y, rgb.z, i, paramVal, (float)(i*(pixWidth+blankWidth)), 0.,0., pixWidth, pixWidth);
    }
  }

  void resetMe(){
    int i;
    for(i=0; i<nItems; i++){
      myRects->at(i)->isInverted = false;
    }
  }
};

class glPixelRect;

class glVideo: public glPlottable{
  public:
  glSlider * mySlider;
  SA<glImage *> * myImages;
  vector<int> frameIndex;
  int loopMode;
  int isPaused;
  vector<int>::iterator ii;
  int iit;
  int iitLast;
  SA< vector< glPixelRect *> > * imagePointsOnPath;
  int * randomFramePointer;

  glVideo(zprInstance * parent, vector<glImage *> * _myFrames, int * _randomFramePointer, glSlider * mySlide, SA< vector< glPixelRect *> > * imagePointsOnPath_);

  void drawMe();
  int nFrames();
  void rebuffer();
};

class glPoint: public glPlottable{
  public:
  SA<float> * p;
  int N;
  vec3 x;
  glPoint(zprInstance * parent, float X, float Y, float Z, float R, float G, float B){
    x.init(X,Y,Z); rgb.init(R,G,B);
    initName(parent,false);
    addPointToBoundingBox(X,Y,Z);

  }
  void drawMe(){
    colorMe();
    glPushMatrix();
    glPointSize(2.);
    glBegin(GL_POINTS);
    x.vertex();
    glEnd();
    glPopMatrix();
  }
};

class glLine: public glPlottable{
  public:

  vec3 x1, x2;
  int myWidth;
  glLine( zprInstance * parent, vec3 & a, vec3 & b, int R, int G, int B){
    x1.init(a); x2.init(b);
    rgb.init(R,G,B);
    myWidth = 1.;
    initName(parent, false);
    addPointToBoundingBox(x1.x, x1.y, x1.z);
    addPointToBoundingBox(x2.x, x2.y, x2.z);
  }

  void setWidth( int w){
    myWidth = w;
  }

  void drawMe(){
    colorMe();
    glLineWidth(myWidth);
    glPushMatrix();
    glBegin(GL_LINES);
    x1.vertex();
    x2.vertex();
    glEnd();
    glPopMatrix();
  }
};

class glBasicSphere: public glPlottable{
  public:
  float size; int circles, stacks;

  glBasicSphere(){
    setRelativePosition = 0;
  }

  glBasicSphere(zprInstance * parent, float X, float Y, float Z, float R, float G, float B, float Size, int Circles, int Stacks){
    x.init(X,Y,Z);
    setRelativePosition = 0;
    init(parent, X, Y, Z, R, G, B, Size, Circles, Stacks);
    addPointToBoundingBox(X,Y,Z);
    myLinks.clear();
    isLinkd=false;
  }

  glBasicSphere(int _myLabel, zprInstance * parent, float X, float Y, float Z, float R, float G, float B, float Size, int Circles, int Stacks){
    x.init(X,Y,Z);
    setRelativePosition = 0;
    isLabelled = true; myLabel = _myLabel;
    init(parent, X, Y, Z, R, G, B, Size, Circles, Stacks);
    addPointToBoundingBox(X,Y,Z);
    myLinks.clear();
    isLinkd=false;
  }

  void init(zprInstance * parent, float X, float Y, float Z, float R, float G, float B, float Size, int Circles, int Stacks){
    x.init(X,Y,Z); rgb.init(R,G,B);
    size=Size; circles=Circles; stacks=Stacks;
    initName(parent,true);
    addPointToBoundingBox(X,Y,Z);
    setRelativePosition = 0;
    myLinks.clear();
    isLinkd=false;
  }
  void drawMe(int highlight){
    colorMe(highlight);
    if(isPicked() && (setRelativePosition==1)){
      parentZprInstance->rX = x;
    }
    glPushMatrix();
    vec3 tx(x-parentZprInstance->rX);
    glTranslatef(tx.x, tx.y, tx.z);
    glPushName(myName);
    glutSolidSphere(size, circles, stacks);
    glPopName();
    glPopMatrix();
  }

  void drawMe(){
    colorMe();
    if(isPicked() && (setRelativePosition==1)){
      parentZprInstance->rX = x;
    }
    glPushMatrix();
    vec3 tx(x-parentZprInstance->rX);
    glTranslatef(tx.x, tx.y, tx.z);
    glPushName(myName);
    glutSolidSphere(size, circles, stacks);
    glPopName();
    glPopMatrix();
  }
};

class glClustSphere: public glBasicSphere{
  public:
  int b1;
  int b2;
  int b3;
  SA<float> * myDat;
  myImg * img;
  glClustSphere(zprInstance * parent, SA<float> * thedat, myImg * imga, int i){

    myDat = NULL;
    img = NULL;
    myDat = thedat;
    img = imga;
    myBi = parent->myBi;
    b1 = (*myBi)[0];
    b2 = (*myBi)[1];
    b3 = (*myBi)[2];
    float rr = (*myDat)[b1];
    float gg = (*myDat)[b2];
    float bb = (*myDat)[b3];

    x.init(rr,gg,bb);
    rgb.init(rr,gg,bb);

    size=0.1;//Size;
    circles=10;//Circles;
    stacks=10;//Stacks;
    setRelativePosition = 0;//1;
    setLabel(i);
    initName(parent,true);
    addPointToBoundingBox(rr,gg,bb);
    parentZprInstance = parent;
    setRelativePosition = 0;

  }

  void setXYZ(float xMin, float xMax, int NRow, int NCol, int p1i, int p1j, float p1d){
    float x1z = max( (float) NRow, (float) NCol ) * (p1d - xMin) / (abs(xMax-xMin));
    float x1x = (float)( p1i);
    float x1y = (float)( p1j);
    x.init(x1x,x1y,x1z); rgb.init(x1x,x1y,x1z);
  }

  void rebuffer(){
    b1 = (*myBi)[0];
    b2 = (*myBi)[1];
    b3 = (*myBi)[2];
    float rr = (*myDat)[b1];
    float gg = (*myDat)[b2];
    float bb = (*myDat)[b3];
    x.init(rr,gg,bb); rgb.init(rr,gg,bb);
    addPointToBoundingBox(rr,gg,bb);
  }

  void drawMe();

};

class glArrow: public glPlottable{
  public:
  vec3 x1;
  vec3 x2;

  glArrow(zprInstance * parent, float rr, float gg, float bb){
    initName(parent,false);
    rgb.init(rr,gg,bb);
  }

  void setXYZ(float x1x, float x1y, float x1z, float x2x, float x2y, float x2z){
    x1.init(x1x,x1y,x1z);
    x2.init(x2x,x2y,x2z);
    addPointToBoundingBox(x1.x,x1.y,x1.z);
    addPointToBoundingBox(x2.x,x2.y,x2.z);
  }

  void setXYZ_Scaled(float x1x, float x1y, float x1z, float x2x, float x2y, float x2z, float xMin, float xMax, int NRow, int NCol){
    x1.init(x1x,x1y, max((float)NRow,(float)NCol ) * (x1z - xMin) / (abs(xMax-xMin)));
    x2.init(x2x,x2y, max((float)NRow,(float)NCol ) * (x2z - xMin) / (abs(xMax-xMin)));
    addPointToBoundingBox(x1.x,x1.y,x1.z);
    addPointToBoundingBox(x2.x,x2.y,x2.z);
  }

  void setXYZ(float xMin, float xMax, int NRow, int NCol, int p1i, int p2i, int p1j, int p2j,float p1d, float p2d){
    x1.z = max( (float) NRow, (float) NCol ) * (p1d - xMin) / (abs(xMax-xMin));
    x2.z = max( (float) NRow, (float) NCol ) * (p2d - xMin) / (abs(xMax-xMin));
    x1.x = (float)( p1i);
    x2.x = (float)( p2i);
    x1.y = (float)( p1j);
    x2.y = (float)( p2j);
    addPointToBoundingBox(x1.x,x1.y,x1.z);
    addPointToBoundingBox(x2.x,x2.y,x2.z);
  }

  void drawMe(){
    vec3 Mx1(x1-(parentZprInstance->rX));
    vec3 Mx2(x2-(parentZprInstance->rX));

    vec3 dx(Mx2-Mx1);
    float len = dx.length();
    float tPL = ARROWHEAD_LENGTH;
    vec3 tx( dx - (dx*(tPL/len)));
    vec3 normalV( -dx.y, dx.x, 0.);
    normalV = normalV / normalV.length();
    float tNormal = ARROWHEAD_WIDTH;
    vec3 leftP( tx + ( normalV*tNormal));
    vec3 rightP( tx - ( normalV*tNormal));

    vec3 nV2( tx.cross(normalV));
    nV2 = nV2 / nV2.length();
    vec3 leftP2( tx + ( nV2*tNormal));
    vec3 rightP2( tx - ( nV2*tNormal));

    glPushName(myName);
    colorMe();
    glLineWidth(1.5);

    glPushMatrix();

    glBegin(GL_LINES); Mx1.vertex(); Mx2.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); leftP.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); rightP.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); leftP2.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); rightP2.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPopName();
  }

  void drawMe(int highlight){
    vec3 Mx1(x1-(parentZprInstance->rX));
    vec3 Mx2(x2-(parentZprInstance->rX));
    vec3 dx(Mx2-Mx1);
    float len = dx.length();
    float tPL = ARROWHEAD_LENGTH;
    vec3 tx( dx - (dx*(tPL/len)));
    vec3 normalV( -dx.y, dx.x, 0.);
    normalV = normalV / normalV.length();
    float tNormal = ARROWHEAD_WIDTH;
    vec3 leftP( tx + ( normalV*tNormal));
    vec3 rightP( tx - ( normalV*tNormal));

    vec3 nV2( tx.cross(normalV));
    nV2 = nV2 / nV2.length();
    vec3 leftP2( tx + ( nV2*tNormal));
    vec3 rightP2( tx - ( nV2*tNormal));

    glPushName(myName);
    colorMe(highlight);
    glLineWidth(1.5);

    glPushMatrix();
    glBegin(GL_LINES); Mx1.vertex(); Mx2.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); leftP.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); rightP.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); leftP2.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(Mx1.x, Mx1.y, Mx1.z);
    glBegin(GL_TRIANGLES); tx.vertex(); rightP2.vertex(); dx.vertex(); glEnd();
    glPopMatrix();

    glPopName();
  }

};

#endif //#ifndef NEWZPR_H

//Based on Nigel's work below:

/*
* Zoom-pan-rotate mouse manipulation module for GLUT
* Version 0.4, October 2003
*
* Nigel Stewart
* School of Computer Science and Information Technology
* RMIT University
* nigels@cs.rmit.edu.au
*
* Instructions
* ------------
*
* Call zprInit() immediately after your call to glutCreateWindow()
*
* The ZPR module handles glutReshapeFunc(), glutMouseFunc() and glutMotionFunc()
* Applications should not bypass the ZPR handlers for reshape or mouse events.
*
* Mouse manipulation of the GLUT window via the modelview matrix:
*
* Left button -> rotate
* Middle button -> zoom
* Right button -> pan
*
* Picking is also provided via two configurable callbacks:
*
* void zprSelectionFunc(void (*f)(void))
*
* The draw function to be called in OpenGL selection
* mode in response to a mouse-down button event.
*
* void zprPickFunc(void (*f)(GLint name))
*
* The callback function which will receive the
* top-most item of the name stack of the closest selection
* hit. If there is no selection hit, -1
*
* Limitations
* -----------
*
* Works best with zprReferencePoint appropriately configured.
* Works best with ortho projection.
* You may need to use glEnable(GL_NORMALIZATION) for correct lighting.
* Near and far clip planes are hard-coded.
* Zooming and rotation is centered on the origin.
* Only one window can use the callbacks at one time.
*
*/

#endif