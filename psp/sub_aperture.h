/*******************************************************************************

File     : sub_aperture.h
Project  : ESA_POLSARPRO
Authors  : Laurent FERRO-FAMIL
Version  : 1.0
Creation : 04/2005
Update   :

*-------------------------------------------------------------------------------
INSTITUT D'ELECTRONIQUE et de TELECOMMUNICATIONS de RENNES (I.E.T.R)
UMR CNRS 6164
Groupe Image et Teledetection
Equipe SAPHIR (SAr Polarimetrie Holographie Interferometrie Radargrammetrie)
UNIVERSITE DE RENNES I
P�le Micro-Ondes Radar
B�t. 11D - Campus de Beaulieu
263 Avenue G�n�ral Leclerc
35042 RENNES Cedex
Tel :(+33) 2 23 23 57 63
Fax :(+33) 2 23 23 69 63
e-mail : eric.pottier@univ-rennes1.fr, laurent.ferro-famil@univ-rennes1.fr
*-------------------------------------------------------------------------------
Description :  SUB APERTURE Routines
*-------------------------------------------------------------------------------
Routines    :
void write_config_sub(char *dir, int Nlig, int Ncol, char *PolarCase, char *PolarType, int nim, int Nsub_im, float pct, float squint);
void hamming(float ham_a,float *ham_win,int n);
void estimate_spectrum(FILE *in_file[],int Npolar,float **spectrum, float **fft_im,int Nlig,int Ncol,int N,int Naz,int Nrg,int AzimutFlag);
void correction_function(int Npolar,float **spectrum,float **correc,int weight,int *lim1,int *lim2,int N,int N_smooth);
void compensate_spectrum(FILE *in_file,float *correc,float **fft_im,int Nlig,int Ncol,int N,int Naz,int Nrg,int AzimutFlag);
void select_sub_spectrum(float **fft_im,float **c_im,int offset,float *ham_win,int n_ham,float *vec1,int N,int Nrg);

*******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#ifdef _WIN32
#include <dos.h>
#include <conio.h>
#endif

#ifndef FlagSubAperture
#define FlagSubAperture

#include "matrix.h"
#include "processing.h"
#include "util.h"

void write_config_sub(char *dir, int Nlig, int Ncol, char *PolarCase, char *PolarType, int nim, int Nsub_im, float pct, float squint);
void hamming(float ham_a,float *ham_win,int n);
void estimate_spectrum(FILE *in_file[],int Npolar,float **spectrum, float **fft_im,int Nlig,int Ncol,int N,int Naz,int Nrg,int AzimutFlag);
void correction_function(int Npolar,float **spectrum,float **correc,int weight,int *lim1,int *lim2,int N,int N_smooth);
void compensate_spectrum(FILE *in_file,float *correc,float **fft_im,int Nlig,int Ncol,int N,int Naz,int Nrg,int AzimutFlag);
void select_sub_spectrum(float **fft_im,float **c_im,int offset,float *ham_win,int n_ham,float *vec1,int N,int Nrg);

#endif
