#include"misc.h"
#include<stdio.h>
#include<stdlib.h>
#include<memory.h>
#define conv_type double // assume double as intermediary for type conversions

/* lookup table: ENVI image type -> c/c++ type
1 -> 8-bit byte
2 -> 16-bit signed integer
3 -> 32-bit signed long integer
4 -> 32-bit floating point
5 -> 64-bit double-precision floating point
12 -> 16-bit unsigned integer
13 -> 32-bit unsigned long integer
14 -> 64-bit signed long integer
15 -> 64-bit unsigned long integer */

void check(const char * type, int result, int expected){
  if(result != expected){
    printf("Error: %s size = %d, expected %d\n", type, result, expected);
    exit(1);
  }
}

int main(int argc, char ** argv){
  check("char", sizeof(char), 1);
  check("short signed int", sizeof(short signed int), 2);
  check("signed int", sizeof(signed int), 4);
  check("float", sizeof(float), 4);
  check("double", sizeof(double), 8);
  check("short unsigned int", sizeof(short unsigned int), 2);
  check("unsigned int", sizeof(unsigned int), 4);
  check("long signed int", sizeof(long signed int), 8);
  check("long unsigned int", sizeof(long unsigned int), 8);

  if(argc != 4){
    printf("cv.cpp: Convert between ENVI image data types Ash Richardson 200806 with updates 200905;\n");
    printf("\treimplemented 20170617 and updated 20200226\n");

    printf("Usage: cv [input file name] [output file name] [output data type]\n");
    printf("Supported Data Types:\n1=8-bit byte\n2=16-bit signed integer\n3=32-bit signed long integer\n4=32-bit floating point\n5=64-bit double-precision floating point\n12=16-bit unsigned integer\n13=32-bit unsigned long integer\n14=64-bit signed long integer\n15=64-bit unsigned long integer\n\nUnsupported types:\n6=2x32-bit complex, real-imaginary pair of double precision\n9=2x64-bit double-precision complex, real-imaginary pair of double precision\n");
    printf("\nNote: this program is not affected by interleave type (although it naively processes the file on a band by band basis).\n");
    printf("\nIMPORTANT: program assumes intel convention (byte order 0).\n");
    exit(1);
  }

  char * infilename = argv[1]; // input file name

  size_t ns, nl, nb; 
  int idatatype = -1; // atoi(argv[2]); // data type
  int samples = -1; // atoi(argv[3]); // image dimensions
  int lines = -1; // atoi(argv[4]);
  int bands = -1; //atoi(argv[5]);
  char * outfilename = argv[2]; // output file name
  int odatatype = atoi(argv[3]); // output data type
  
  str ifn(infilename);
  str hfn(hdr_fn(ifn));
  idatatype = (int) hread(hfn, nl, ns, nb);
  samples = (int) ns;
  lines = (int) nl;
  bands = (int) nb; // clean up the redundant variables later

  FILE * infile; FILE * outfile; // file handles
  long int n_pix = lines * samples; // number of pixels

  infile = fopen(infilename, "r");  // open files and check
  outfile = fopen(outfilename, "w");
  if(!infile){
    printf("Error: could not open input file\n"); exit(1);
  }
  if(!outfile){
    printf("Error: could not open input file\n"); exit(1);
  }

  /* data variables */
  char * dat_1; // char 
  char * out_1;
  short signed int * dat_2; // short signed int
  short signed int * out_2;
  signed int * dat_3; // signed int
  signed int * out_3;
  float * dat_4; // float
  float * out_4;
  double * dat_5; // double
  double * out_5;
  short unsigned int * dat_12; // short unsigned int
  short unsigned int * out_12;
  unsigned int * dat_13;  // unsigned int
  unsigned int * out_13;
  long signed int * dat_14; // long signed int
  long signed int * out_14;
  unsigned long int * dat_15; // long unsigned int
  unsigned long int * out_15;

  switch(idatatype){
    case 1:
    dat_1 = (char *) malloc(n_pix * sizeof(char));
    if(!dat_1){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 2:
    dat_2 = (short signed int *) malloc(n_pix * sizeof(short signed int));
    if(!dat_2){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 3:
    dat_3 = (signed int *) malloc(n_pix * sizeof(signed int));
    if(!dat_3){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 4:
    dat_4 = (float *) malloc(n_pix * sizeof(float));
    if(!dat_4){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 5:
    dat_5 = (double *) malloc(n_pix * sizeof(double));
    if(!dat_5){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 12:
    dat_12 = (short unsigned int *) malloc(n_pix * sizeof(short unsigned int));
    if(!dat_12){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 13:
    dat_13 = (unsigned int *) malloc(n_pix * sizeof(unsigned int));
    if(!dat_13){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 14:
    dat_14 = (long signed int *) malloc(n_pix * sizeof(long signed int));
    if(!dat_14){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 15:
    dat_15 = (long unsigned int *) malloc(n_pix * sizeof(long unsigned int));
    if(!dat_15){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    default:
    printf("Error: Disallowed input data type value\n");
    exit(1);
    break;
  }

  switch(odatatype){
    case 1:
    out_1 = (char *) malloc(n_pix * sizeof(char));
    if(!out_1){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 2:
    out_2 = (short signed int *) malloc(n_pix * sizeof(short signed int));
    if(!out_2){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 3:
    out_3 = (signed int *) malloc(n_pix * sizeof(signed int));
    if(!out_3){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 4:
    out_4 = (float *) malloc(n_pix * sizeof(float));
    if(!out_4){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 5:
    out_5 = (double *) malloc(n_pix * sizeof(double));
    if(!out_5){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 12:
    out_12 = (short unsigned int *) malloc(n_pix * sizeof(short unsigned int));
    if(!out_12){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 13:
    out_13 = (unsigned int *) malloc(n_pix * sizeof(unsigned int));
    if(!out_13){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 14:
    out_14 = (long signed int *) malloc(n_pix * sizeof(long signed int));
    if(!out_14){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    case 15:
    out_15 = (long unsigned int *) malloc(n_pix * sizeof(long unsigned int));
    if(!out_15){
      printf("Malloc failure\n"); exit(1);
    }
    break;
    default:
    printf("Error: Disallowed input data type value\n"); exit(1); break;
  }

  double * tmp = (conv_type *) malloc(n_pix * sizeof(conv_type));

  if(!tmp){
    printf("Malloc failed.\n");
    exit(1);
  }

  register int i, j;

  for(i = 0; i < bands; i++){
    printf("\rBand %i of %i", i + 1, bands);

    switch(idatatype){
      case 1: fread(dat_1, n_pix * sizeof(char), 1, infile); break;
      case 2: fread(dat_2, n_pix * sizeof(short signed int), 1, infile); break;
      case 3: fread(dat_3, n_pix * sizeof(signed int), 1, infile); break;
      case 4: fread(dat_4, n_pix * sizeof(float), 1, infile); break;
      case 5: fread(dat_5, n_pix * sizeof(double), 1, infile); break;
      case 12: fread(dat_12, n_pix * sizeof(short unsigned int), 1, infile); break;
      case 13: fread(dat_13, n_pix * sizeof(unsigned int), 1, infile); break;
      case 14: fread(dat_14, n_pix * sizeof(long signed int), 1, infile); break;
      case 15: fread(dat_15, n_pix * sizeof(long unsigned int), 1, infile); break;
      default: printf("Error: Disallowed input data type value\n"); exit(1); break;
    }

    switch(idatatype){
      case 1: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_1[j];} break;
      case 2: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_2[j];} break;
      case 3: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_3[j];} break;
      case 4: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_4[j];} break;
      case 5: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_5[j];} break;
      case 12: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_12[j];} break;
      case 13: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_13[j];} break;
      case 14: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_14[j];} break;
      case 15: for(j = 0; j < n_pix; j++){tmp[j] = (conv_type)dat_15[j];} break;
      default: printf("Error: Disallowed input outa type value\n"); exit(1); break;
    }

    switch(odatatype){
      case 1: for(j = 0; j < n_pix; j++) out_1[j] = (char)tmp[j]; break;
      case 2: for(j = 0; j < n_pix; j++) out_2[j] = (short unsigned int)tmp[j]; break;
      case 3: for(j = 0; j < n_pix; j++) out_3[j] = (signed int)tmp[j]; break;
      case 4: for(j = 0; j < n_pix; j++) out_4[j] = (float)tmp[j]; break;
      case 5: for(j = 0; j < n_pix; j++) out_5[j] = (double)tmp[j]; break;
      case 12: for(j = 0; j < n_pix; j++) out_12[j] = (short unsigned int)tmp[j]; break;
      case 13: for(j = 0; j < n_pix; j++) out_13[j] = (unsigned int)tmp[j]; break;
      case 14: for(j = 0; j < n_pix; j++) out_14[j] = (long signed int)tmp[j]; break;
      case 15: for(j = 0; j < n_pix; j++) out_15[j] = (long unsigned int)tmp[j]; break;
      default: printf("Error: Disallowed input outa type value\n"); exit(1); break;
    }

    switch(odatatype){
      case 1: fwrite(out_1, n_pix * sizeof(char), 1, outfile); break;
      case 2: fwrite(out_2, n_pix * sizeof(short signed int), 1, outfile); break;
      case 3: fwrite(out_3, n_pix * sizeof(signed int), 1, outfile); break;
      case 4: fwrite(out_4, n_pix * sizeof(float), 1, outfile); break;
      case 5: fwrite(out_5, n_pix * sizeof(double), 1, outfile); break;
      case 12: fwrite(out_12, n_pix * sizeof(short unsigned int), 1, outfile); break;
      case 13: fwrite(out_13, n_pix * sizeof(unsigned int), 1, outfile); break;
      case 14: fwrite(out_14, n_pix * sizeof(long signed int), 1, outfile); break;
      case 15: fwrite(out_15, n_pix * sizeof(long unsigned int), 1, outfile); break;
      default: printf("Error: Disallowed input outa type value\n"); exit(1); break;
    }
  }

  fclose(infile);
  fclose(outfile);
  return 0;
}
