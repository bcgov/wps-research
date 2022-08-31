#pragma once
#ifndef NEWZPR_H
#define NEWZPR_H
//NEED SINGLETON CLASS FOR ZPR module (to assign functions (using array/ vector of function pointers))

#ifdef WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
# include <OpenGL/gl.h>
# include <OpenGL/glu.h>
# include <GLUT/glut.h>
#else
# include <GL/gl.h>
# include <GL/glu.h>
# include <GL/glut.h>
#endif

#include <stdio.h>
#include <set>
#include <fstream>
#include <iostream>

using namespace std;

extern int WINDOWX;
extern int WINDOWY;

extern double _left;
extern double _right;
extern double _top;
extern double _bottom;
extern double _near;
extern double _far;
extern double _zNear;
extern double _zFar;
extern std::set<GLint> myPickNames;

extern int _mouseX;// = 0;
extern int _mouseY;// = 0;
extern int _mouseLeft;// = false;
extern int _mouseMiddle;// = false;
extern int _mouseRight;// = false;

extern double _dragPosX;// = 0.0;
extern double _dragPosY;// = 0.0;
extern double _dragPosZ;// = 0.0;

extern double _matrix[16];
extern double _matrixInverse[16];

void zprInit();

static GLfloat zprReferencePoint[4];

/* Picking API (Optional) */

extern void zprSelectionFunc(void (*f)(void)); /* Selection-mode draw function */
extern void zprPickFunc(void (*f)(GLint name)); /* Pick event handling function */

extern void (*pick)(GLint name);//= NULL;

#endif

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