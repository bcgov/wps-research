/* 20220830 dem.cpp: plot an image in 3d

usage:
  dem [input raster file] [z coordinate band] [r coord band] [g coord band] [b coord band] 

20220831 add left-right arrows to shift z band, up-down arrows to shift r,g,b bands
*/
#define Z_SCALE 0.25
int ri, gi, bi, zi;
#define MYFONT GLUT_BITMAP_HELVETICA_12
#define STR_MAX 1000
#include"newzpr.h"
#include"pthread.h"
#include"time.h"
#include"vec3d.h"
#include"misc.h"
#include<stdio.h>
#include<stdlib.h>
#include<cfloat>

vector<string> band_names;
size_t np;

vec3d rX; // 3d reference point
char console_string[STR_MAX];
size_t nrow, ncol, nband;  // base image data

float * dat, zmax, zmin; // dem data: nrow * ncol linear array of floats
float * rgb; // basemap data

vec3d * points; // 3d points for visualization

void setOrthographicProjection() {
  int h = WINDOWX;// glutGet(GLUT_SCREEN_HEIGHT);
  int w = WINDOWY; //glutGet(GLUT_SCREEN_WIDTH);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0., (float)w, 0., (float)h);
  glScalef(1., -1., 1.);
  glTranslatef(0, -1.*(float)h, 0);
  glMatrixMode(GL_MODELVIEW);
}

void resetPerspectiveProjection(){
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

/* convert world xyz to screen xyz */
void world2screen(float x, float y, float z, float & screenX, float & screenY, float & screenZ){
    double vx, vy, vz;
    GLint view[4];
    GLdouble proj[16];
    GLdouble model[16];
    glGetDoublev(GL_PROJECTION_MATRIX,proj);
    glGetDoublev(GL_MODELVIEW_MATRIX,model);
    glGetIntegerv(GL_VIEWPORT,view);
    gluProject(x, y, z, model, proj, view, &vx, &vy, &vz);
    screenY = (float)(vy - glutGet(GLUT_WINDOW_HEIGHT));
    (screenX = (float)vx), (screenZ = (float)vz);
}

void renderBitmapString(float x, float y, void *font, const char *string){
  const char *c;
  glRasterPos2f(x,y);
  for (c=string; *c != '\0'; c++)
    glutBitmapCharacter(font, *c);
}

void drawText(char * s, int offset){
  glColor3f(0.0f,1.0f,0.0f);
  setOrthographicProjection();
  glPushMatrix();
  glLoadIdentity();
  int lightingState = glIsEnabled(GL_LIGHTING);
  glDisable(GL_LIGHTING);
  renderBitmapString(3, WINDOWY - 3 - offset, (void *)MYFONT, (const char *)s); 
  if(lightingState) glEnable(GL_LIGHTING);
  glPopMatrix();
  resetPerspectiveProjection();
}

GLint selected; // picking?
int special_key; // 114

/* Draw axes */
#define STARTX 500
#define STARTY 500
int fullscreen;
clock_t start_time;
clock_t stop_time;
#define SECONDS_PAUSE 0.4
int console_position;
int renderflag;

void _pick(GLint name){
  if(myPickNames.size() > 0){
    cout << "PickSet:";
    std::set<GLint>::iterator it;
    for(it = myPickNames.begin(); it != myPickNames.end(); it++)
      cout << *it << "," ;
    cout << endl;
    fflush(stdout);
  }
}

float a1, a2, a3;

class point{
  public:
  point(){
  }
};

/* Callback function for drawing */
void display(void){
  glPointSize(2.);
  size_t i, j, k;
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();
  glTranslatef(-rX.x, -rX.y, -rX.z);

  if(true) { //false){
    vec3d X; // axes
    X.x = X.y = X.z = 0; X.axis(.01);
    X.x = X.y = 1;       X.axis(.01);
    X.x = 1; X.y = 0;    X.axis(.01);
    X.x = 0; X.y = 1;    X.axis(.01);
    rX.axis(.1);
  }

  // iterate over dataset
  GLint picked = -1;
  float R, G, B, x;
  size_t np = nrow * ncol;

  for0(i, nrow){
    for0(j, ncol){
      k = i * ncol + j;
      /*x = (dat[k] - zmin) / (zmax - zmin);
      (R = x), (G = 0.), (B = 1. - x);
     */
      R = dat[ri * np + k];
      G = dat[gi * np + k];
      B = dat[bi * np + k];

      //   (R = rgb[k]), (G = rgb[np + k]), (B = rgb[np + np + k]);
      if(!(isnan(R) || isnan(G) || isnan(B))){
        glColor3f(R, G, B);

          // plot as poly, if we can
          if(i + 1 < nrow && j + 1 < ncol){
            size_t i_1 = (i + 1) * ncol + j;
            glBegin(GL_POLYGON);
            (points[k].vertex()), (points[i_1].vertex());
            (points[i_1 + 1].vertex()), (points[i_1 - ncol + 1].vertex());
            glEnd();
          }
      }
    }
  }

  glPopMatrix();
  glutSwapBuffers();
  renderflag = false;
}

/* Callback function for pick-event handling from ZPR */

void quitme(){
  exit(0);
}

void special(int key, int x, int y){
  printf("special %d\n", key);
  special_key = key;
}

/* Keyboard functions */
void keyboard(unsigned char key, int x, int y){
  switch(key){
    case 8 :
    case 127:
    
      if(console_position > 0){
        console_position --;
        console_string[console_position]='\0';
        printf("STRING: %s\n", &console_string[0]);
        display();
      }
      break;

    // Enter
    case 13 :
      if(true){
        //printf( "%d Pressed RETURN\n",(char)key);
        str S(console_string);
        cout << "[" << S << "]" << endl;
 
	long int x;

	str title(str("z=(") + band_names[zi] + str(") r=(") + band_names[ri] + str(") g=(") + band_names[gi] + str(") b=(") + band_names[bi] + str(")"));

	if(console_string[0] == 'r' || console_string[0] == 'g' || console_string[0] == 'b' || console_string[0] == 'z'){
	  x = atol(str(&console_string[1]).c_str());
	  if(console_string[0] == 'r'){
	    if(x >= 0 && x < nband) ri = x;
	  }
	  if(console_string[0] == 'g'){
	    if(x >= 0 && x < nband) gi = x;
	  }
    	  if(console_string[0] == 'b'){
	    if(x >= 0 && x < nband) bi = x;
	  }
  	  if(console_string[0] == 'z'){
	    if(x >= 0 && x < nband){
	      zi = x;
	      size_t i, j, k;
  	      for0(i, nrow){
   	        for0(j, ncol){
      	          k = (i * ncol) + j;
	          points[k].z = dat[k + (zi * np)];
   	          points[k].z *= Z_SCALE;
    	        }
	      }
            } 
	  }

	  str title(str("z=(") + band_names[zi] + str(") r=(") + band_names[ri] + str(") g=(") + band_names[gi] + str(") b=(") + band_names[bi] + str(")"));
	  glutSetWindowTitle(title.c_str());
	}

	console_string[0]='\0';
        console_position=0;
        display();
      }
      break;

    case GLUT_KEY_LEFT:
      {
	zi -= 1;
	if(zi < 0){
	  zi = nband - 1;
        }
      }
      break;
    case GLUT_KEY_RIGHT:
      {
        zi += 1;
        if(zi >= nband){
          zi = 0;
        }
      }

      break;
    case GLUT_KEY_UP:
      break;
    case GLUT_KEY_DOWN:
      break;


    // Escape
    case 27 :
      quitme();
      exit(0); //printf( "%d Pressed Esc\n",(char)key);
      break;

    default:  //printf( "Pressed key %c AKA %d at position %d % d\n",(char)key, key, x, y);
      console_string[console_position++] = (char)key;
      console_string[console_position]='\0';
      printf("STRING: %s\n", &console_string[0]);
      display();
      break;
  }
}

static GLfloat light_ambient[] = {
  0.0, 0.0, 0.0, 1.0 };
static GLfloat light_diffuse[] = {
  1.0, 1.0, 1.0, 1.0 };
static GLfloat light_specular[] = {
  1.0, 1.0, 1.0, 1.0 };
static GLfloat light_position[] = {
  1.0, 1.0, 1.0, 0.0 };

static GLfloat mat_ambient[] = {
  0.7, 0.7, 0.7, 1.0 };
static GLfloat mat_diffuse[] = {
  0.8, 0.8, 0.8, 1.0 };
static GLfloat mat_specular[] = {
  1.0, 1.0, 1.0, 1.0 };
static GLfloat high_shininess[] = {
  100.0 };

// https://computing.llnl.gov/tutorials/pthreads/
void idle(){
  if(renderflag){
    glFlush();
    glutPostRedisplay();
  }
}

int main(int argc, char ** argv){
  ri = gi = bi = 0;
  //==========================================================================
  /* DEM data file */
  str fn(argv[1]);
  
  if(!exists(fn)) err("please check input file");

  /*str fn2(fn + str("_scale.bin"));

    if(!exists(fn2)){
      int a = system((str("raster_scale ") + fn).c_str());
    }
  */
  fn = fn2;
  str hfn(hdr_fn(fn));

  size_t i, j;
  rX.x = rX.y = rX.z = 0.;

  band_names = vector<string>();

  hread(hfn, nrow, ncol, nband, band_names); // load DEM data
  dat = bread(fn, nrow, ncol, nband);
  np = nrow * ncol;
  
  //dont forget to keep aspect ratio!
  zmax = -(float)FLT_MAX;  // Note: not FLT_MIN !!
  zmin = +(float)FLT_MAX;

  /*label = falloc(np);
  for0(i, np) label[i] = 0.;
*/
  // should plot the neighbourhoods from the other direction (climb on -z, too!)
  points = new vec3d[nrow * ncol];
  for0(i, nrow){
    for0(j, ncol){
      size_t k = (i * ncol) + j;
      points[k].x = ((float)j) / ((float)nrow); //- .5;
      points[k].y = 1. - ((float)i) / ((float)nrow); //ncol); //- .5;
      points[k].z = dat[k]; // / 300. ;

      /*
      if(points[k].z > zmax) zmax = points[k].z;
      if(points[k].z < zmin) zmin = points[k].z;
      */
//      printf("x,y,z %f %f %f\n", points[k].x, points[k].y, dat[k]);
    }
  }

  printf("zmin %f zmax %f\n", zmin, zmax);

  /* DEM scaling ? */
  for0(i, nrow){
    for0(j, ncol){
      size_t k = (i * ncol) + j;
      //points[k].z -= zmin;
      //points[k].z /= (zmax - zmin);
      points[k].z *= Z_SCALE; 
    }
  }

  /* centre point */
  rX.x = points[ncol/2 + (nrow/2) * ncol].x;
  rX.y = points[ncol/2 + (nrow/2) * ncol].y;
  rX.z = points[ncol/2 + (nrow/2) * ncol].z;
  
  pick = _pick;
  special_key = selected = -1;
  renderflag = false;
  a1 = a2 = a3 = 1;
  console_position = 0;
  fullscreen = 0;

  str title(str("z=(") + band_names[zi] + str(") r=(") + band_names[ri] + str(") g=(") + band_names[gi] + str(") b=(") + band_names[bi] + str(")"));

  /* Initialise GLUT & create window */
  printf("glutInit()\n");
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(STARTX,STARTY);
  glutCreateWindow(title.c_str());
  zprInit();
  printf("glutCreateWindow()\n");

  /* Configure GLUT callback functions */
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(special); // glutKeyboardUpFunc(keyboardup);
  glutIdleFunc(idle);
  glScalef(0.25,0.25,0.25);
  
  /* Configure ZPR module */
  zprSelectionFunc(display); /* Selection mode draw function */
  zprPickFunc(pick); /* Pick event client callback */

  /* Initialise OpenGL */
  glDepthFunc(GL_LESS);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glEnable(GL_COLOR_MATERIAL);
  glutMainLoop(); // enter GLUT event loop
  return 0;
}
