#pragma once
#ifndef HEADER_vec3d_H
#define HEADER_vec3d_H
#include<cmath>
#include<math.h>

class vec3d{
  public:
  float x, y, z;
  vec3d(float xx, float yy, float zz){
    x = xx; y = yy; z = zz;
  }

  vec3d(vec3d * a){
    x=a->x; y=a->y; z=a->z;
  }

  vec3d(){
  };

  void init(float X, float Y, float Z){
    x=X; y=Y; z=Z;
  }

  void init(vec3d a){
    x=a.x; y=a.y; z=a.z;
  }

  vec3d(const vec3d & other){
    x = other.x; y = other.y; z = other.z;
  }

  void operator=(vec3d rhs){
    x = rhs.x; y = rhs.y; z = rhs.z;
  }

  float dot(vec3d other){
    return x*other.x + y*other.y + z*other.z;
  }

  vec3d cross(vec3d other){
    vec3d ret(y*other.z -z*other.y,
    z*other.x -x*other.z,
    x*other.y -y*other.x );
    return ret;
  }

  int operator==(vec3d rhs){
    return(x==rhs.x && y==rhs.y && z==rhs.z);
  }

  vec3d operator+(vec3d rhs){
    vec3d ret(x + rhs.x, y+rhs.y, z+rhs.z);
    return ret;
  }

  vec3d operator-(vec3d rhs){
    vec3d ret( x-rhs.x, y-rhs.y, z-rhs.z);
    return ret;
  }

  vec3d operator-=(vec3d rhs){
    x -=rhs.x; y -=rhs.y; z -=rhs.z;
    return *this;
  }

  vec3d operator+=(vec3d rhs){
    x +=rhs.x; y +=rhs.y; z +=rhs.z;
    return *this;
  }

  vec3d operator+(float s){
    vec3d ret( x+s,y+s,z+s);
    return ret;
  }

  vec3d operator-(float s){
    vec3d ret( x-s,y-s,z-s);
    return ret;
  }

  vec3d operator*(float s){
    vec3d ret( x*s, y*s, z*s);
    return ret;
  }

  vec3d operator/(float s){
    vec3d ret( x/s, y/s ,z/s);return ret;
  }

  float length(){
    return sqrt( x*x +y*y +z*z);
  }

  inline void vertex(){
    glVertex3f( x, y, z);
  }

  inline void vertex2f(){
    glVertex2f(x,y);
  }

  inline void color(){
    glColor3f(x,y,z);
  }

  inline void axis(float L){
	  // rgb = xyz
    glColor3f(1, 0, 0);
    glBegin(GL_LINES);
    glVertex3f(x, y, z);
    glVertex3f(x+L, y, z);
    glEnd();

    glColor3f(0, 1, 0);
    glBegin(GL_LINES);
    glVertex3f(x, y, z);
    glVertex3f(x, y+L, z);
    glEnd();

    glColor3f(0, 0, 1);
    glBegin(GL_LINES);
    glVertex3f(x, y, z);
    glVertex3f(x, y, z+L);
    glEnd();
  }
};

std::ostream& operator << (std::ostream& os, const vec3d & v){
  os << "[";
  os << v.x << "," << v.y << "," << v.z;
  os << "]";
  return os;
}


#endif
