/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere.. */
#include "my_math.h"
float sgn(float x){
  if(x<0){
    return (-1.);
  }
  else if( x==0){
    return 0.;
  }
  else{
    return 1.;
  };
}

float max( float x, float y){
  return(x > y ? x : y);
}

float min( float x, float y){
  return(x > y ? y : x);
}

float square(float x){
  return(x * x);
}

void ijScreen(float & iScreen, float &jScreen, int i, int j, int NRow, int NCol){
  iScreen = ((float)j)/((float)NCol);
  jScreen = ((float)(NRow - i - 1) /((float)NRow));
}

void ijScreen2(float & iScreen, float &jScreen, int i, int j, int NRow, int NCol){
  iScreen = ((float)j);
  jScreen = (float)(NRow-i-1);
}

float scaleF(float z, float xMin, float xMax, int NRow, int NCol){
  return(max((float)NRow, (float)NCol ) * (z - xMin) / (abs(xMax - xMin)));
}