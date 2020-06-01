/* m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere.. */

#include "image.h"
void myImg::init(char * filename, int nrow, int ncol, int nb){
  NRow = nrow; NCol = ncol; NBand = nb;
  dat.init(nrow*ncol*nb);
  FILE * f = fopen(filename,"rb");

  if(!f){
    dprintf("Error: could not open file: %s", filename);
    exit(1);
  }

  int bytesread = fread( &dat[0], sizeof(float), nrow*ncol*nb, f);
  dprintf("Read %d bytes / %d floats / %d rows / %d cols / %d bands from %s",
  bytesread, nrow*ncol*nb, nrow, ncol, nb, filename);
  fclose(f);

  int ci = 0;
  int i, j;

  float_buffers = new SA< SA<float> * >((my_int)nb);
  printf("\n");
  for(i = 0; i < nb; i++){
    SA<float> * newB = new SA<float>((my_int)0);
    newB->mySize = NRow*NCol;
    newB->sizeI = NRow;
    newB->sizeJ = NCol;
    newB->sizeK = 0;
    newB->elements = &(dat.elements)[i*NRow*NCol];
    //scaleband(newB); //scale into range 0-1.
    float_buffers->at(i) = newB;
  }
  printf("\n");
  myBi = new SA<int>(3);
  for(i = 0; i < 3; i++){
    (*myBi)[i]=i;
  }
  //set default band visualization to 1,2,3;
}
myImg::myImg(myImg * other){
  NRow = other->NRow; NCol = other->NCol; NBand = other->NBand;
  dat.mySize = other->dat.mySize;
  dat.sizeI = other->dat.sizeI; dat.sizeJ = other->dat.sizeJ; dat.sizeK = other->dat.sizeK;
  dat.elements = &((other->dat.elements)[0]);
  float_buffers = other->float_buffers;
}
void myImg::scaleband( SA<float> * buf){
  //this scaling needs to be fixed. zeroing data isn't quite OK.
  float MAX = buf->max();
  float MIN = buf->min();
  // printf(" MIN %e MAX %e \n", MIN, MAX);
  size_t i = 0;
  float dat;
  for(i = 0; i < buf->size(); i++){
    dat = buf->at(i);
    buf->at(i) = (dat-MIN) / (MAX-MIN);
    dat = buf->at(i);
    if(isnan(dat) || isinf( dat)){
      buf->at(i) = FLT_EPSILON;
    }
  }
  MAX = buf->max();
  MIN = buf->min();
}

void myImg::printAsciiCSV(char * ASCIIFILE){
  FILE * afile = fopen(ASCIIFILE, "wb");
  if(!afile){
    dprintf("Error: could not open file: %s\n", ASCIIFILE);
    exit(1);
  }
  int kk, i;
  for(kk = 0; kk < NRow * NCol; kk++){
    for(i = 0; i < NBand; i++){
      fprintf(afile, "%7.7e,", (float_buffers->at(i))->at(kk));
    }
    fprintf(afile, "\n");
  }
  fclose(afile);
}
