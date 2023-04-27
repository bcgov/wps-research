/* GLUT menu example from https://www.openglprojects.in/2016/12/opengl-glut-create-menu-example.html
Modified 20230426. To compile and run:
	g++ test.cpp -lglut -lGL  -o test
	./test
*/
#include <GL/glut.h>
static int window, returnmenu, returnsubmenu, returnsubmenucolor1, returnsubmenucolor2, returnsubmenucolor3, returnsubmenucolor4, value = 0;

void menu(int n){
  if (n == 0){
    glutDestroyWindow(window);
    exit(0);
  }
  else{
    value = n;
  }
  glutPostRedisplay();
}

void createMenu(void){
  //Teapot
  returnsubmenucolor1 = glutCreateMenu(menu);
  glutAddMenuEntry("Red", 11);
  glutAddMenuEntry("White", 12);
  glutAddMenuEntry("Yellow", 13);
  glutAddMenuEntry("Green", 14);
  glutAddMenuEntry("Blue", 15);

  //Sphere
  returnsubmenucolor2 = glutCreateMenu(menu);
  glutAddMenuEntry("Red", 21);
  glutAddMenuEntry("White", 22);
  glutAddMenuEntry("Yellow", 23);
  glutAddMenuEntry("Green", 24);
  glutAddMenuEntry("Blue", 25);

  //Torus
  returnsubmenucolor3 = glutCreateMenu(menu);
  glutAddMenuEntry("Red", 31);
  glutAddMenuEntry("White", 32);
  glutAddMenuEntry("Yellow", 33);
  glutAddMenuEntry("Green", 34);
  glutAddMenuEntry("Blue", 35);

  //Cone
  returnsubmenucolor4 = glutCreateMenu(menu);
  glutAddMenuEntry("Red", 41);
  glutAddMenuEntry("White", 42);
  glutAddMenuEntry("Yellow", 43);
  glutAddMenuEntry("Green", 44);
  glutAddMenuEntry("Blue", 45);

  returnmenu = glutCreateMenu(menu); //function to call menu function and return value
  glutAddMenuEntry("Clear", 1);
  glutAddSubMenu("Teapot", returnsubmenucolor1);
  glutAddSubMenu("Sphere", returnsubmenucolor2);
  glutAddSubMenu("Torus", returnsubmenucolor3);
  glutAddSubMenu("Cone", returnsubmenucolor4);
  glutAddMenuEntry("Quit", 0);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}
void display(void){
  glClear(GL_COLOR_BUFFER_BIT);
  if (value == 1){
    return;
  }
  else if (value == 11){
    glPushMatrix();
    glColor3d(1.0, 0.0, 0.0);
    glutSolidTeapot(0.5);
    glPopMatrix();
  }
  else if (value == 12){
    glPushMatrix();
    glColor3d(1.0, 1.0, 1.0);
    glutSolidTeapot(0.5);
    glPopMatrix();
  }
  else if (value == 13){
    glPushMatrix();
    glColor3d(1.0, 1.0, 0.0);
    glutSolidTeapot(0.5);
    glPopMatrix();
  }
  else if (value == 14){
    glPushMatrix();
    glColor3d(0.0, 1.0, 0.0);
    glutSolidTeapot(0.5);
    glPopMatrix();
  }
  else if (value == 15){
    glPushMatrix();
    glColor3d(0.0, 0.0, 1.0);
    glutSolidTeapot(0.5);
    glPopMatrix();
  }
  else if (value == 21){
    glPushMatrix();
    glColor3d(1.0, 0.0, 0.0);
    glutWireSphere(0.5, 50, 50);
    glPopMatrix();
  }
  else if (value == 22){
    glPushMatrix();
    glColor3d(1.0, 1.0, 1.0);
    glutWireSphere(0.5, 50, 50);
    glPopMatrix();
  }
  else if (value == 23){
    glPushMatrix();
    glColor3d(1.0, 1.0, 0.0);
    glutWireSphere(0.5, 50, 50);
    glPopMatrix();
  }
  else if (value == 24){
    glPushMatrix();
    glColor3d(0.0, 1.0, 0.0);
    glutWireSphere(0.5, 50, 50);
    glPopMatrix();
  }
  else if (value == 25){
    glPushMatrix();
    glColor3d(0.0, 0.0, 1.0);
    glutWireSphere(0.5, 50, 50);
    glPopMatrix();
  }
  else if (value == 31){
    glPushMatrix();
    glColor3d(1.0, 0.0, 0.0);
    glutWireTorus(0.3, 0.6, 100, 100);
    glPopMatrix();
  }
  else if (value == 32){
    glPushMatrix();
    glColor3d(1.0, 1.0, 1.0);
    glutWireTorus(0.3, 0.6, 100, 100);
    glPopMatrix();
  }
  else if (value == 33){
    glPushMatrix();
    glColor3d(1.0, 1.0, 0.0);
    glutWireTorus(0.3, 0.6, 100, 100);
    glPopMatrix();
  }
  else if (value == 34){
    glPushMatrix();
    glColor3d(0.0, 1.0, 0.0);
    glutWireTorus(0.3, 0.6, 100, 100);
    glPopMatrix();
  }
  else if (value == 35){
    glPushMatrix();
    glColor3d(0.0, 0.0, 1.0);
    glutWireTorus(0.3, 0.6, 100, 100);
    glPopMatrix();
  }
  else if (value == 41){
    glPushMatrix();
    glColor3d(1.0, 0.0, 0.0);
    glRotated(65, -1.0, 0.0, 0.0);
    glutWireCone(0.5, 1.0, 50, 50);
    glPopMatrix();
  }
  else if (value == 42){
    glPushMatrix();
    glColor3d(1.0, 1.0, 1.0);
    glRotated(65, -1.0, 0.0, 0.0);
    glutWireCone(0.5, 1.0, 50, 50);
    glPopMatrix();
  }
  else if (value == 43){
    glPushMatrix();
    glColor3d(1.0, 1.0, 0.0);
    glRotated(65, -1.0, 0.0, 0.0);
    glutWireCone(0.5, 1.0, 50, 50);
    glPopMatrix();
  }
  else if (value == 44){
    glPushMatrix();
    glColor3d(0.0, 1.0, 0.0);
    glRotated(65, -1.0, 0.0, 0.0);
    glutWireCone(0.5, 1.0, 50, 50);
    glPopMatrix();
  }
  else if (value == 45){
    glPushMatrix();
    glColor3d(0.0, 0.0, 1.0);
    glRotated(65, -1.0, 0.0, 0.0);
    glutWireCone(0.5, 1.0, 50, 50);
    glPopMatrix();
  }
  glFlush();
}

int main(int argc, char **argv){
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
  glutInitWindowSize(500, 500);
  glutInitWindowPosition(100, 100);
  window = glutCreateWindow("Menu");
  createMenu();
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glutDisplayFunc(display);
  glutMainLoop();
  return 0;
}
