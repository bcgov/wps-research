/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere.. */
#ifndef __IMAGE_H
#define __IMAGE_H
#pragma once
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>
#include <memory.h>
#include <math.h>
#include <float.h>
#include "SA.h"
using namespace std;

class myImg{
  public:
  SA<int> * myBi;

  int getBi(int i){
    return myBi->at(i);
  }

  void assignProjection( SA<int> * myb){
    myBi = myb;
  }

  myImg(){
  }

  myImg(myImg * other); //this version just makes a shadow.
  void init(const char * filename, size_t nrow, size_t ncol, size_t nb);

  void initSameSize(myImg * other){
    NRow = other->NRow;
    NCol = other->NCol;
    NBand = other->NBand;
    dat.init(NRow * NCol * NBand);
    size_t nb = other->NBand;
    float_buffers = new SA< SA<float> * >( (my_int) nb );
    printf("\n"); size_t i;
    for(i = 0; i < nb; i++){
      SA<float> * newB = new SA<float>((my_int)0);
      newB->mySize = NRow*NCol;
      newB->sizeI = NRow;
      newB->sizeJ = NCol;
      newB->sizeK = 0;
      newB->elements = (&(dat.elements)[i*NRow*NCol]);
      scaleband(newB); //scale into range 0-1.
      float_buffers->at(i) = newB;
    }
    printf("\n");
    for(i = 0; i < 3; i++){
      myBi->at(i) = other->getBi(i);
    }
  }
  void initFrom(SA<float> * other, size_t nr, size_t nc, size_t nb){
    NRow = nr; NCol = nc; NBand = nb;
    dat.init(NRow*NCol*NBand);
    dat.sizeI = NRow; dat.sizeJ = NCol; dat.sizeK = NBand;
    size_t i;
    for(i = 0; i < other->size(); i++){
      dat[i] = (*other)[i];
    }
    float_buffers = new SA< SA<float> * >( (my_int) nb );
    printf("\n");
    for(i = 0; i < nb; i++){
      SA<float> * newB = new SA<float>((my_int) 0);
      newB->mySize = NRow*NCol;
      newB->sizeI = NRow;
      newB->sizeJ = NCol;
      newB->sizeK = 0;
      newB->elements = (&(dat.elements)[i*NRow*NCol]);
      scaleband(newB); //scale into range 0-1.
      float_buffers->at(i) = newB;
    }
    printf("\n");
    myBi = new SA<int>(3);
    for(i = 0; i < 3; i++){
      myBi->at(i) = i;
    }
  }

  void initBlank(size_t nr, size_t nc, size_t nb){
    NRow = nr; NCol = nc; NBand = nb;
    dat.init(NRow*NCol*NBand);
    dat.sizeI = NRow; dat.sizeJ = NCol; dat.sizeK = NBand;
    size_t i;
    float_buffers = new SA< SA<float> *>(nb);
    for(i = 0; i < nb; i++){
      SA<float> * newB = new SA<float>(0);
      newB->mySize = NRow * NCol;
      newB->sizeI = NRow;
      newB->sizeJ = NCol;
      newB->sizeK = 0;
      newB->elements = (&(dat.elements)[i*NRow*NCol]);
      float_buffers->at(i) = newB;
    }
    myBi = new SA<int>(3);
    for(i = 0; i < 3; i++){
      myBi->at(i) = i;
    }

  }

  // init from other, with offset

  void clear(){
    long int i;
    for(i = 0; i < mySize(); i++){
      (*this)[i] =0.;
    }
  }
  int mySize(){
    return NRow*NCol*NBand;
  }
  void assignFrom(myImg * other){
    if(other->mySize() != this->mySize()){
      dprintf("Error: myImg.h: tried to assign from array of different size.");
    }
    size_t i;
    for(i = 0; i < other->mySize(); i++){
      (*this)[i] = (*other)[i];
    }
    return;
  }
  void assignFrom(SA<float> * other){
    if(other->size() != this->mySize()){
      dprintf("Error: myImg.h: tried to assign from array of different size.");
    }
    size_t i;
    for(i = 0; i < other->size(); i++){
      (*this)[i] = (*other)[i];
    }
    return;
  }

  myImg(char * filename, int nrow, int ncol, int nb){
    init(filename,nrow,ncol,nb);
  }
  SA< SA<float> * > * getFloatBuffers(){
    return float_buffers;
  }

  size_t NRow, NCol, NBand;
  SA<float> dat;

  private:
  SA< SA<float> * > * float_buffers;
  void scaleband( SA<float> * buf);
  void printAsciiCSV(char * ASCIIFILE);

  public:
  inline float & operator[](my_int subscript){
    return dat[subscript];
  }
};
#endif
