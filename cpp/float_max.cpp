/* Find the largest successive float that's intra-convertible with integer..
.. the number of "classes" that could be represented using a float type */
#include<stdio.h>
#include<stdlib.h>
int main(int argc, char ** argv){
  float x = 0.;
  size_t y = 0;

  while(((float)y == x) &&
        ((size_t)x == y)){
    x ++;
    y ++;
  }
  printf("%zu\n", y - 1);
  return 0;
}
