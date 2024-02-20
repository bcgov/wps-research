#pragma once

/******************************************************************************

   File     : util.h
   Project  : ESA_POLSARPRO
   Authors  : Eric POTTIER, Laurent FERRO-FAMIL
   Version  : 1.0
   Creation : 09/2003
   Update   :

*-------------------------------------------------------------------------------
   INSTITUT D'ELECTRONIQUE et de TELECOMMUNICATIONS de RENNES (I.E.T.R)
   UMR CNRS 6164                                                                                          
   Groupe Image et Teledetection
   Equipe SAPHIR (SAr Polarimetrie Holographie Interferometrie Radargrammetrie)
   UNIVERSITE DE RENNES I
   Pôle Micro-Ondes Radar
   Bât. 11D - Campus de Beaulieu
   263 Avenue Général Leclerc
   35042 RENNES Cedex
   Tel :(+33) 2 23 23 57 63
   Fax :(+33) 2 23 23 69 63
   e-mail : eric.pottier@univ-rennes1.fr, laurent.ferro-famil@univ-rennes1.fr
*-------------------------------------------------------------------------------
   Description :  UTIL Routines
*-------------------------------------------------------------------------------
   Routines    :
struct Pix
struct Pix *Create_Pix(struct Pix *P, float x,float y);
struct Pix *Remove_Pix(struct Pix *P_top, struct Pix *P);
float my_round(float v);
void edit_error(char *s1,char *s2)
void check_file(char *file);
void check_dir(char *dir);
void read_config(char *dir, int *Nlig, int *Ncol, char *PolarCase, char *PolarType);
void write_config(char *dir, int Nlig, int Ncol, char *PolarCase, char *PolarType);
void my_randomize (void);
float my_random (float num);
float my_eps_random (void);
struct cplx;
cplx    cadd(cplx a,cplx b);
cplx    csub(cplx a,cplx b);
cplx    cmul(cplx a,cplx b);
cplx    cdiv(cplx a,cplx b);
cplx    cconj(cplx a);
float   cimg(cplx a);
float   crel(cplx a);
float   cmod(cplx a);
float   cmod2(cplx a);
float   angle(cplx a);
cplx    cplx_sinc(cplx a);

*******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#ifdef _WIN32
#include <dos.h>
#include <conio.h>
#endif

#ifndef FlagUtil
#define FlagUtil

#define eps 1.E-15
#define pi 3.14159265358979323846
#define M_C 299792458 

#define INIT_MINMAX 1.E+30

/* Return nonzero value if X is not +-Inf or NaN.  */
#define my_isfinite(x) (sizeof (x) == sizeof (float) ? 1 : 0)

struct Pix
{
  float x;
  float y;
  struct Pix *next;
};

struct Pix *Create_Pix (struct Pix *P, float x, float y);
struct Pix *Remove_Pix (struct Pix *P_top, struct Pix *P);

float my_round (float v);

void edit_error (char *s1, char *s2);

void check_file (char *file);
void check_dir (char *dir);

void read_config (char *dir, int *Nlig, int *Ncol, char *PolarCase,
		  char *PolarType);
void write_config (char *dir, int Nlig, int Ncol, char *PolarCase,
		   char *PolarType);

void my_randomize (void);
float my_random (float num);
float my_eps_random (void);

typedef struct
{
	float re;
	float im;
}cplx;

cplx    cadd(cplx a,cplx b);
cplx    csub(cplx a,cplx b);
cplx    cmul(cplx a,cplx b);
cplx    cdiv(cplx a,cplx b);
cplx    cconj(cplx a);
float   cimg(cplx a);
float   crel(cplx a);
float   cmod(cplx a);
float   cmod2(cplx a);
float   angle(cplx a);
cplx    cplx_sinc(cplx a);

#endif



#include "util.c"

