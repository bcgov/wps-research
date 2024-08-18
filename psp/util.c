#pragma once

/*******************************************************************************
PolSARpro v4.0 is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 (1991) of the License, or any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. 

See the GNU General Public License (Version 2, 1991) for more details.

********************************************************************************

File     : util.c
Project  : ESA_POLSARPRO
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Version  : 1.0
Creation : 09/2003
Update   : 12/2006 (Stephane MERIC)

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
#include <time.h>

#ifdef _WIN32
#include <dos.h>
#include <conio.h>
#endif

#include "graphics.h"
#include "matrix.h"
#include "processing.h"
#include "util.h"

/*******************************************************************************
Structure: Create_Pix
Authors  : Laurent FERRO-FAMIL
Creation : 07/2003
Update   :
*-------------------------------------------------------------------------------
Description :
*-------------------------------------------------------------------------------
Inputs arguments :

Returned values  :

*******************************************************************************/
struct Pix *Create_Pix(struct Pix *P, float x, float y)
{
    if (P == NULL) {
	P = (struct Pix *) malloc(sizeof(struct Pix));
	P->x = x;
	P->y = y;
	P->next = NULL;
    } else
	edit_error("Error Create Pix", "");
    return P;
}

/*******************************************************************************
Structure: Remove_Pix
Authors  : Laurent FERRO-FAMIL
Creation : 07/2003
Update   :
*-------------------------------------------------------------------------------
Description :
*-------------------------------------------------------------------------------
Inputs arguments :

Returned values  :

*******************************************************************************/
struct Pix *Remove_Pix(struct Pix *P_top, struct Pix *P)
{
    struct Pix *P_current;

    if (P == NULL)
	edit_error("Error Create Pix", "");
    if (P == P_top) {
	P_current = P_top;
	P = P->next;
	free((struct Pix *) P_current);
    } else {
	if (P->next == NULL) {
	    P_current = P_top;
	    while (P_current->next != P)
		P_current = P_current->next;
	    P = P_current;
	    P_current = P_current->next;
	    free((struct Pix *) P_current);

	} else {
	    P_current = P_top;
	    while (P_current->next != P)
		P_current = P_current->next;
	    P = P_current;
	    P_current = P_current->next;
	    P->next = P_current->next;
	    free((struct Pix *) P_current);
	}
    }
    P_current = NULL;
    return P;
}

/*******************************************************************************
Routine  : my_round
Authors  : Laurent FERRO-FAMIL
Creation : 07/2003
Update   : 12/2006 (Stephane MERIC)
*-------------------------------------------------------------------------------
Description : Round function
*-------------------------------------------------------------------------------
Inputs arguments :

Returned values  :

*******************************************************************************/
float my_round(float v)
{
#if defined(__sun) || defined(__sun__)
    static inline float floorf (float x) { return floor (x);}
#endif

#ifndef _WIN32
    return (floorf(v + 0.5));
#endif
#ifdef _WIN32
    return (floor(v + 0.5));
#endif

}

/*******************************************************************************
Routine  : edit_error
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Displays an error message and exits the program
*-------------------------------------------------------------------------------
Inputs arguments :
s1    : message to be displayed
s2    : message to be displayed
Returned values  :
void
*******************************************************************************/
void edit_error(char *s1, char *s2)
{
    printf("\n A processing error occured ! \n %s%s\n", s1, s2);
    exit(1);
}

/*******************************************************************************
Routine  : check_file
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Checks and corrects slashes in file string
*-------------------------------------------------------------------------------
Inputs arguments :
file    : string to be checked
Returned values  :
void
*******************************************************************************/
void check_file(char *file)
{
#ifdef _WIN32
    int i;
    i = 0;
    while (file[i] != '\0') {
	if (file[i] == '/')
	    file[i] = '\\';
	i++;
    }
#endif
}

/*******************************************************************************
Routine  : check_dir
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Checks and corrects slashes in directory string
*-------------------------------------------------------------------------------
Inputs arguments :
dir    : string to be checked
Returned values  :
void
*******************************************************************************/
void check_dir(char *dir)
{
#ifndef _WIN32
    strcat(dir, "/");
#else
    int i;
    i = 0;
    while (dir[i] != '\0') {
	if (dir[i] == '/')
	    dir[i] = '\\';
	i++;
    }
    strcat(dir, "\\");
#endif
}

/*******************************************************************************
Routine  : read_config
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Read a configuration file
*-------------------------------------------------------------------------------
Inputs arguments :
dir    : location of the config.txt file
Returned values  :
Nlig   : Read number of lines
Ncol   : Read number of rows
PolarCase : Monostatic / Bistatic
PolarType : Full / PP1 / PP2 / PP3 / PP4
*******************************************************************************/
void
read_config(char *dir, int *Nlig, int *Ncol, char *PolarCase,
	    char *PolarType)
{
    char file_name[1024];
    char Tmp[1024];
    FILE *file;

    sprintf(file_name, "%sconfig.txt", dir);
    if ((file = fopen(file_name, "r")) == NULL)
	edit_error("Could not open configuration file : ", file_name);


    fscanf(file, "%s\n", Tmp);
    fscanf(file, "%i\n", &*Nlig);
    fscanf(file, "%s\n", Tmp);
    fscanf(file, "%s\n", Tmp);
    fscanf(file, "%i\n", &*Ncol);
    fscanf(file, "%s\n", Tmp);
    fscanf(file, "%s\n", Tmp);
    fscanf(file, "%s\n", PolarCase);
    fscanf(file, "%s\n", Tmp);
    fscanf(file, "%s\n", Tmp);
    fscanf(file, "%s\n", PolarType);

    fclose(file);
}

/*******************************************************************************
Routine  : write_config
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Writes a configuration file
*-------------------------------------------------------------------------------
Inputs arguments :
dir    : location of the config.txt file
Nlig   : number of lines to be written
Ncol   : number of rows to be written
PolarCase : Monostatic / Bistatic
PolarType : Full / PP1 / PP2 / PP3 / PP4
Returned values  :
void
*******************************************************************************/
void
write_config(char *dir, int Nlig, int Ncol, char *PolarCase,
	     char *PolarType)
{
    char file_name[1024];
    FILE *file;

    sprintf(file_name, "%sconfig.txt", dir);
    if ((file = fopen(file_name, "w")) == NULL)
	edit_error("Could not open configuration file : ", file_name);

    fprintf(file, "Nrow\n");
    fprintf(file, "%i\n", Nlig);
    fprintf(file, "---------\n");
    fprintf(file, "Ncol\n");
    fprintf(file, "%i\n", Ncol);
    fprintf(file, "---------\n");
    fprintf(file, "PolarCase\n");
    fprintf(file, "%s\n", PolarCase);
    fprintf(file, "---------\n");
    fprintf(file, "PolarType\n");
    fprintf(file, "%s\n", PolarType);
    fclose(file);
}

/*******************************************************************************
Routine  : my_randomize
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Initialisation of the Random generator
*-------------------------------------------------------------------------------
Inputs arguments :
Returned values  :
*******************************************************************************/
void my_randomize(void)
{
    srand((unsigned) time(NULL));
}

/*******************************************************************************
Routine  : my_random
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Random value
*-------------------------------------------------------------------------------
Inputs arguments :
Returned values  :
float Random value
*******************************************************************************/
float my_random(float num)
{
    float res;
    res = (float) ((rand() * num) / (RAND_MAX + 1.0));
    return (res);
}

/*******************************************************************************
Routine  : my_eps_random
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  eps/20 < Random value < 19*eps / 20
*-------------------------------------------------------------------------------
Inputs arguments :
Returned values  :
float Random value
*******************************************************************************/
float my_eps_random(void)
{
    float res;
    res = (float) ((rand() * 1.0) / (RAND_MAX + 1.0));
    res = eps * (1. + 9. * res) / 10.;
    res = res - eps / 20.;
    return (res);
}

/*******************************************************************************
Routine  : cconj
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Complex Conjugate
*-------------------------------------------------------------------------------
*******************************************************************************/
cplx cconj(cplx a)
{
 cplx res;

 res.re= a.re;
 res.im=-a.im;

 return(res);
}

/*******************************************************************************
Routine  : cadd
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Complex Addition
*-------------------------------------------------------------------------------
*******************************************************************************/
cplx cadd(cplx a,cplx b)
{
 cplx res;

 res.re=a.re+b.re;
 res.im=a.im+b.im;

 return(res);
}

/*******************************************************************************
Routine  : csub
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Complex Substraction
*-------------------------------------------------------------------------------
*******************************************************************************/
cplx csub(cplx a,cplx b)
{
 cplx res;

 res.re=a.re-b.re;
 res.im=a.im-b.im;

 return(res);
}

/*******************************************************************************
Routine  : cmul
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Complex Multiplication
*-------------------------------------------------------------------------------
*******************************************************************************/
cplx cmul(cplx a,cplx b)
{
 cplx res;

 res.re=a.re*b.re-a.im*b.im;
 res.im=a.re*b.im+a.im*b.re;

 return(res);
}

/*******************************************************************************
Routine  : cdiv
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Complex Division
*-------------------------------------------------------------------------------
*******************************************************************************/
cplx cdiv(cplx a,cplx b)
{
 cplx res;
 float val;

 if ((val=cmod(b))<eps)
 {
  b.re = eps;
  b.im = eps;
 }  

 res=cmul(a,cconj(b));
 res.re=res.re /(val*val);
 res.im=res.im /(val*val);

 return(res);
}

/*******************************************************************************
Routine  : cimg
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Imaginary Part of a Complex Number
*-------------------------------------------------------------------------------
*******************************************************************************/
float cimg(cplx a)
{
 return(a.im);
}

/*******************************************************************************
Routine  : crel
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Real Part of a Complex Number
*-------------------------------------------------------------------------------
*******************************************************************************/
float crel(cplx a)
{
 return(a.re);
}

/*******************************************************************************
Routine  : cmod
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Complex Modulus
*-------------------------------------------------------------------------------
*******************************************************************************/
float cmod(cplx a)
{
 float res;

 res=sqrt(a.re*a.re+a.im*a.im);

 return(res);
}

/*******************************************************************************
Routine  : cmod2
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Complex Modulus
*-------------------------------------------------------------------------------
*******************************************************************************/
float cmod2(cplx a)
{
 float res;

 res=(a.re*a.re+a.im*a.im);

 return(res);
}

/*******************************************************************************
Routine  : angle
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description :  Argument of a Complex
*-------------------------------------------------------------------------------
*******************************************************************************/
float angle(cplx a)
{
 float ang;

 if((fabs(a.re)<eps)&&(fabs(a.im)<eps))
  ang=0;
 else
  ang=atan2(a.im,a.re);

 return(ang);
}

/*******************************************************************************
Routine  : cplx_sinc
Authors  : Laurent FERRO-FAMIL
Creation : 08/2005
Update   :
*-------------------------------------------------------------------------------
Description : Sinc Function of a Complex
*-------------------------------------------------------------------------------
*******************************************************************************/
cplx cplx_sinc(cplx a)
{
  cplx cplx_dum1,cplx_dum2,res;

  cplx_dum1.re= exp(-a.im)*cos(a.re); cplx_dum1.im= exp(-a.im)*sin(a.re);
  cplx_dum2.re= exp(a.im)*cos(a.re); cplx_dum2.im= -exp(a.im)*sin(a.re);

  cplx_dum1= csub(cplx_dum1,cplx_dum2);
  cplx_dum2.re= 0.0; cplx_dum2.im= -1.0/2.0;

  cplx_dum1= cmul(cplx_dum1,cplx_dum2);

  res= cdiv(cplx_dum1,a);

  return(res);
}

