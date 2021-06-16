/* based on m3ta3 */
#ifndef __SA_H
#define __SA_H
#pragma once
#include <vector>
#include <math.h>
#include <float.h>
#include <fstream>
#include <ostream>
#include <memory.h>
#include <iostream>
#include <algorithm>

using namespace std;

using std::ostream;
using std::vector;

#define my_int int

#define dprintf(...) \
(printf("\n\tDPRINTF %s(%d): [", __FILE__, __LINE__), printf(__VA_ARGS__), printf("]\n"))

static int isGood(float x){
  return !(isnan(x) || isinf(x));
}

template<class T> class SA{
  public:

  T * elements;
  my_int mySize;
  my_int sizeI, sizeJ, sizeK;

  inline void clear(void){
    memset(elements, '\0', mySize * sizeof(T));
  }

  void init(my_int size){
    if(elements && mySize != size){
      // free(elements);
    }
    elements = NULL;
    mySize = size;
    if(mySize > 0){
      elements = NULL;
      elements = new T[mySize];
      if(!elements){
        dprintf("Error (SA.h): Array allocation failure.\n");
        exit(1);
      }
      memset(elements, '\0', mySize * sizeof(T));
    }
  }

  inline void init(my_int sizei, my_int sizej){
    init(sizei * sizej);
    sizeI = sizei;
    sizeJ = sizej;
    sizeK = 0;
  }

  void init(my_int sizei, my_int sizej, my_int sizek){
    init(sizei * sizej * sizek);
    sizeI = sizei;
    sizeJ = sizej;
    sizeK = sizek;
  }

  SA(){
    elements = NULL;
    init(0);
  }

  SA(my_int size){
    elements = NULL;
    init(size);
    sizeI = mySize;
    sizeJ = 0;
    sizeK = 0;
  }

  SA(my_int isize, my_int jsize){
    elements = NULL;
    init(isize * jsize);
    sizeI = isize;
    sizeJ = jsize;
    sizeK = 0;
  }

  SA(my_int isize, my_int jsize, my_int ksize){
    elements = NULL;
    init(isize * jsize * ksize);
    sizeI = isize;
    sizeJ = jsize;
    sizeK = ksize;
  }

  ~SA(){
    free(elements);
  }

  inline SA(SA<T> * other){
    if(!other){
      mySize = 0;
      elements = NULL;
    }
    mySize = other->size();
    if(mySize == 0){
      elements = NULL;
      return;
    }
    else{
      init(other->size());
      for(register my_int i = 0; i < mySize; i++){
        elements[i] = (*other)[i];
      }
    }
  }

  inline void clearMe(){
    memset(elements, '\0', mySize * sizeof(T));
  }

  inline my_int size(){
    return mySize;
  }

  inline my_int length(){
    return mySize;
  }

  inline T & operator[](my_int subscript){
    if(mySize == 0){
      dprintf("Error (SA.cpp) has size()=0, thus subscript (%d) is out of range", (int)subscript);
      exit(1);
    }
    if((subscript >= mySize) || (subscript < 0)){
      dprintf("Error (SA.cpp): subscript (%d) is out of range", (int)subscript); exit(1);
    }
    return elements[subscript];
  }

  inline T & at(my_int subscript){
    return (*this)[subscript];
  }

  inline T & at(my_int subi, my_int subj){
    if(sizeJ <= 0){
      cerr << "Warning: SA.h: used 2-d indexing on 1-d vector.\n";
    }
    if(sizeK > 0){
      cerr << "Warning: SA.h: used 2-d indexing on 3-d vector.\n";
    }
    return (*this)[(sizeJ * subi) + subj];
  }

  inline T & at(my_int subi, my_int subj, my_int subk){
    if(sizeJ * sizeK <= 0){
      cerr << "Warning: SA.h: used 3-d indexing on non-3-d vector.\n";
    }
    return (*this)[(subk * sizeJ * sizeI) + (sizeJ * subi) + subj];
  }

  inline float max(){
    float max = FLT_MIN;
    int i;
    float d;
    for(i = 0; i < mySize; i++){
      d = ((float)elements[i]);
      if(isGood(d)){
        if(d > max){
          max = d;
        }
      }
    }
    return(max);
  }

  inline float min(){
    float min = FLT_MAX;
    int i;
    float d;
    for(i = 0; i < mySize; i++){
      d = ((float)elements[i]);
      if(isGood(d)){
        if(d < min){
          min = d;
        }
      }
    }
    return(min);
  }
};

template <class T> inline ostream &operator<<(ostream &output, SA<T> &out){
  register my_int i = 0;
  for(i = 0; i < out.length(); i++){
    if(i != out.length() - 1){
      output << out[i] << ",";
    }
    else{
      output << out[i];
    }
  }
  return output;
}
#endif
