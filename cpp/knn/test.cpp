#include<stdio.h>
#include<math.h>
#include<stdlib.h>
int main(int argc, char ** argv){

  float a = .1 / 0.;
  float b = NAN;
  printf("%d %d\n", isinf(a), isinf(-a));
  printf("%d %d\n", isnan(b), isnan(-b));
  return 0;
}
