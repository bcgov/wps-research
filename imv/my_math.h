/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere.. */

#ifndef __MYMATH_H
#define __MYMATH_H
#pragma once

#include <cmath>

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
#endif

using namespace std;

extern float sgn(float x);
extern float max( float x, float y);
extern float min(float x, float y);
extern float square( float x );
extern void ijScreen(float & iScreen, float &jScreen, int i, int j, int NRow, int NCol);
extern void ijScreen2(float & iScreen, float &jScreen, int i, int j, int NRow, int NCol);
extern float scaleF(float z, float xMin, float xMax, int NRow, int NCol);

class vec3{
  public:
  float x, y, z;
  vec3( float xx, float yy, float zz){
    x = xx;
    y = yy;
    z = zz;
  }

  vec3(){
  };

  void init(float X, float Y, float Z){
    x = X;
    y = Y;
    z = Z;
  }
  void init(vec3 & a){
    x = a.x;
    y = a.y;
    z = a.z;
  }

  vec3(const vec3 & other){
    x = other.x;
    y = other.y;
    z = other.z;
  }

  vec3 & operator=(vec3 & rhs){
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
  }

  float dot( vec3 & other){
    return x * other.x + y * other.y + z * other.z;
  }

  vec3 & cross(vec3 & other){
    vec3 * ret = new vec3();
    ret->x = y * other.z - z * other.y;
    ret->y = z * other.x - x * other.z;
    ret->z = x * other.y - y * other.x;
    return *ret;
  }

  int operator==(vec3 rhs){
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  vec3 & operator+(vec3 rhs){
    vec3 * ret = new vec3(x + rhs.x, y+rhs.y, z+rhs.z);
    return *ret;
  }

  vec3 operator-(vec3 rhs){
    vec3* ret = new vec3( x-rhs.x, y-rhs.y, z-rhs.z);
    return *ret;
  }

  vec3 & operator+(float s){
    vec3* ret = new vec3( x+s,y+s,z+s);
    return *ret;
  }

  vec3 & operator-(float s){
    vec3* ret = new vec3( x-s,y-s,z-s);
    return *ret;
  }

  vec3 & operator*(float s){
    vec3* ret = new vec3( x*s, y*s, z*s);
    return *ret;
  }

  vec3 & operator/(float s){
    vec3* ret = new vec3( x/s, y/s ,z/s);
    return *ret;
  }

  float length(){
    return sqrt( x*x +y*y +z*z);
  }

  inline void vertex(){
    glVertex3f(x, y, z);
  }
};

#endif