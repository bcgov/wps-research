/*******************************************************************************
PolSARpro v4.0 is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 (1991) of the License, or any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. 

See the GNU General Public License (Version 2, 1991) for more details.

********************************************************************************

File     : sub_aperture.c
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

#include "graphics.h"
#include "matrix.h"
#include "processing.h"
#include "util.h"
#include "sub_aperture.h"

/*******************************************************************************
   Routine  : write_config
   Authors  : Laurent FERRO-FAMIL
   Creation : 04/2005
   Update   :
*-------------------------------------------------------------------------------
   Description :  Writes an image number of lines and rows from a configuration file
*-------------------------------------------------------------------------------
   Inputs arguments :
      dir       : location of the config.txt file
      Nlig      : number of lines to be written
      Ncol      : number of rows to be written
      PolarCase : Polarimetric Case (Monostatic, Bistatic, Intensities)
      PolarType : Polarimetric Data Type (full, pp1,pp2,pp3,pp5,pp6,pp7)
      nim       : Sub Aperture Number
      Nsub_im   : Total Sub Aperture Number
      pct       : Resolution Fraction
      squint    : FrequencyOffset
   Returned values  :
   void
*******************************************************************************/
void write_config_sub(char *dir, int Nlig, int Ncol, char *PolarCase, char *PolarType, int nim, int Nsub_im, float pct, float squint)
{
 char file_name[1024];
 FILE *file;
 
 sprintf(file_name,"%sconfig.txt",dir);
 if ((file=fopen(file_name,"w"))==NULL)
  edit_error("Could not open configuration file : ",file_name);

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
    fprintf(file, "---------\n");
    fprintf(file, "Sub Aperture Number\n");
    fprintf(file, "%i\n", nim);
    fprintf(file, "---------\n");
    fprintf(file, "Total Sub Aperture Number\n");
    fprintf(file, "%i\n", Nsub_im);
    fprintf(file, "---------\n");
    fprintf(file, "ResolutionFraction(pct)\n");
    fprintf(file, "%7.3f\n",pct);
    fprintf(file, "---------\n");
    fprintf(file, "FrequencyOffset(pct)\n");
    fprintf(file, "%7.3f\n",squint);
    fclose(file);
}
 
/*******************************************************************************
   Routine  : hamming
   Authors  : Laurent FERRO-FAMIL
   Creation : 04/2005
   Update   :
*-------------------------------------------------------------------------------
   Description :  Create a Hamming weighting function
*-------------------------------------------------------------------------------
   Inputs arguments :
      ham_win : Hamming Window
      n       : number of points
   Returned values  :
   void
*******************************************************************************/
void hamming(float ham_a,float *ham_win,int n)
{
 int lig;

 if(!(n%2))
  edit_error("Hamming window width is not an odd number of pixels","");
 
 for(lig=0;lig<n;lig++)   
  ham_win[lig] = ham_a+(1-ham_a)*cos(2*pi*(lig-(n-1)/2)/n);
}

/*******************************************************************************
   Routine  : estimate_spectrum
   Authors  : Laurent FERRO-FAMIL
   Creation : 04/2005
   Update   :
*-------------------------------------------------------------------------------
   Description :  Estimation of the SAR image Doppler spectrum
*-------------------------------------------------------------------------------
   Inputs arguments :
   Returned values  :
   void
*******************************************************************************/
void estimate_spectrum(FILE *in_file[],int Npolar,float **spectrum, float **fft_im,int Nlig,int Ncol,int N,int Naz,int Nrg,int AzimutFlag)
{
  int az,rg,lig,col,np;
  float *M_in;
  
  M_in = vector_float(2*Ncol);
 /* READING */

 for(np=0;np<Npolar;np++)
 {
  /* Read input image */ 
  for (az=0; az<2*N; az++)
   for (rg=0; rg<Nrg; rg++)
    fft_im[rg][az] = 0.;

  for (az=0; az<N; az++) spectrum[np][az] = 0;

  for (lig=0; lig<Nlig; lig++)
  {
  if (lig%(int)(Nlig/20) == 0) {printf("%f\r", 100. * lig * (np+1) / (Nlig - 1) / Npolar);fflush(stdout);}
  fread(&M_in[0],sizeof(float),2*Ncol,in_file[np]);

  if (AzimutFlag == 1)
   {
   /* Transpose the input image to perform the FFT over the lines */   
   for (col=0; col<Ncol; col++)
    {
    fft_im[col][2*lig]   = M_in[2*col];
    fft_im[col][2*lig+1] = M_in[2*col+1];
    }
   }
  else
   {
   for (col=0; col<Ncol; col++)
    {
    fft_im[lig][2*col]   = M_in[2*col];
    fft_im[lig][2*col+1] = M_in[2*col+1];
    }
   }
  } /*end lig */

/* FORWARD FFT AND SPECTRUM AVERAGING IN RANGE*/
  for(rg=0; rg<Nrg; rg++)
  {
   Fft(fft_im[rg],N,+1);
   /* Now sum up in the range direction the amplitudes of the different azimuth spectra*/
   for(az=0; az<N; az++)
    spectrum[np][az] += sqrt(fft_im[rg][2*az]*fft_im[rg][2*az]+fft_im[rg][2*az+1]*fft_im[rg][2*az+1])/Nrg;
  } 
 } /* end np */
}
  
/*******************************************************************************
   Routine  : correction_function
   Authors  : Laurent FERRO-FAMIL
   Creation : 04/2005
   Update   :
*-------------------------------------------------------------------------------
   Description :  Creation of the correction function
*-------------------------------------------------------------------------------
   Inputs arguments :
   Returned values  :
   void
*******************************************************************************/
void correction_function(int Npolar,float **spectrum,float **correc,int weight,int *lim1,int *lim2,int N,int N_smooth)
{
 int az,np,vlim1,vlim2,lim1_max,lim2_min,ii,jj;
 float *vec,max,mean,mean2,cv;
 
 vec = vector_float(N);
 
 /**********************************/
 /*  SPECTRUM LIMITS ESTIMATION    */
 /**********************************/
 lim1_max = -1;
 lim2_min = 2*N;
 for(np=0;np<Npolar;np++)
 {
  /*fft circular shift in order to avoid dicontnuities at zero frequency */
  for(az=0;az<N/2;az++)
  {
   vec[az] = spectrum[np][az+N/2];  vec[az+N/2] = spectrum[np][az];
  } 

  if(weight) 
  {
   vlim1 = 0;
   vlim2 = 0;
   /* If the original spectrum has been weighted for side-lobe suppression,
      estimate the weighting function limits */
   /* Variation coefficient computation for spectrum limits determination */ 
 
   max = 0;
   for(az=(N_smooth-1)/2;az<N-(N_smooth-1)/2;az++)
   {
    /* Reinitialization of the maximum CV value at zero frequency
       in order to detect both lower and upper spectrum limits */
    if(az==N/2) max =0;
    
    /* N_smooth samples sliding averaging filter around the lig offset */
    mean = 0;  mean2  = 0;
    for(ii=-(N_smooth-1)/2;ii<(N_smooth-1)/2+1;ii++)
    {
     mean  += vec[az+ii];
     mean2 += vec[az+ii]*vec[az+ii];
    }
    mean  /=N_smooth;  mean2 /=N_smooth;
    cv = sqrt(mean2-mean*mean)/mean;
    if(cv > max)
    {
     max = cv;
     if(az < N/2) vlim1 = az;
     else vlim2 = az;
    } 
   }
   /*Security Offset*/
   vlim1 += (N_smooth-1)/2;
   vlim2 -= (N_smooth-1)/2;
   /*Store extreme values for lig1, lig2 */
   if(vlim1>lim1_max)
    lim1_max = vlim1;
   if(vlim2<lim2_min)
    lim2_min = vlim2;
  }/*weight*/
  else
  {
   /* Original spectrum has not been weighted for side-lobe suppression*/
   /* Set lim1 and lim2 to arbitrary values*/
   if((*lim2) < 0)
   {
    lim1_max = N_smooth;
    lim2_min = N-N_smooth-1;
   } 
   else
   { 
   /* Set lim1 and lim2 to user-defined values*/
    lim1_max = (*lim1);
    lim2_min = N-(*lim2)-1;
   }
  }/*weight*/
 }/*np*/

 (*lim1) = lim1_max; 
 (*lim2) = lim2_min; 
   
 /*************************************/
 /*  CORRECTION FUNCTION ESTIMATION   */
 /*************************************/
 for(np=0;np<Npolar;np++)
 {
  max = 0; 
 
  for(az=0;az<N/2;az++)
  {
   vec[az] = spectrum[np][az+N/2];  vec[az+N/2] = spectrum[np][az];
  } 
  for(az=1;az<N_smooth;az++)
  {
   vec[(*lim1)-az] = vec[(*lim1)];
   vec[(*lim2)+az] = vec[(*lim2)];
  } 

 /* smoothing from lim1 to lim2 to obtain a better estimate of the spectrum
    amplitude*/
  max = 0; mean = 0;
  az = (*lim1);
  /* Load the averaging filter with the first N_smooth samples */
  for(ii=-(N_smooth-1)/2;ii<(N_smooth-1)/2+1;ii++)
   mean  += vec[az+ii]/N_smooth;
  
  correc[np][az] = mean;

  /* Averaging : Update next and previous values only */
  if(mean > max)   max = mean; 
  for(az=(*lim1)+1;az<(*lim2)+1;az++)
  {
   ii = -1-(N_smooth-1)/2; jj = (N_smooth-1)/2;
   mean  += (vec[az+jj]-vec[az+ii])/N_smooth;
   correc[np][az] = mean;
   if(mean > max) max = mean;
  }   
 /* Normalize the spectrum maximum amplitude to 1 and invert for correction */
  for(az=(*lim1)+1;az<(*lim2)+1;az++) vec[az] = max/correc[np][az];
  for(az=0;az<(*lim1)+1;az++) vec[az]=0;
  for(az=(*lim2)+1;az<N;az++) vec[az]=0;
/* Inverse FFT circular shift */
  for(az=0;az<N/2;az++)
  {
   correc[np][az]     = vec[az+N/2];
   correc[np][az+N/2] = vec[az];
  } 
 }/*np*/ 
} 

/*******************************************************************************
   Routine  : compensate_spectrum
   Authors  : Laurent FERRO-FAMIL
   Creation : 04/2005
   Update   :
*-------------------------------------------------------------------------------
   Description :  determination of the SAR image Doppler compensated spectrum
*-------------------------------------------------------------------------------
   Inputs arguments :
   Returned values  :
   void
*******************************************************************************/
void compensate_spectrum(FILE *in_file,float *correc,float **fft_im,int Nlig,int Ncol,int N,int Naz,int Nrg,int AzimutFlag)
{
 int az,rg,lig,col;
 float *M_in;

 M_in = vector_float(2*Ncol);
 /* Read a polarization channel image */
 rewind(in_file);

 for (az=0; az<2*N; az++)
   for (rg=0; rg<Nrg; rg++)
    fft_im[rg][az] = 0.;

  for (lig=0; lig<Nlig; lig++)
  {
  if (lig%(int)(Nlig/20) == 0) {printf("%f\r", 100. * lig / (Nlig - 1));fflush(stdout);}

  fread(&M_in[0],sizeof(float),2*Ncol,in_file);

  if (AzimutFlag == 1)
   {
   /* Transpose --> Azimuth decomposition */
   for (col=0; col<Ncol; col++)
    {
    fft_im[col][2*lig]   = M_in[2*col];
    fft_im[col][2*lig+1] = M_in[2*col+1];
    }
   }
  else
   {
   for (col=0; col<Ncol; col++)
    {
    fft_im[lig][2*col]   = M_in[2*col];
    fft_im[lig][2*col+1] = M_in[2*col+1];
    }
   }
  } /*end lig */

/* Compute the FFT */
  for(rg=0; rg<Nrg; rg++)
  {
   Fft(fft_im[rg],N,+1);
   for(az=0; az<N; az++)
    {
    fft_im[rg][2*az] *= correc[az];
    fft_im[rg][2*az+1] *= correc[az];
    }
  } 

}    

/*******************************************************************************
   Routine  : select_sub_spectrum
   Authors  : Laurent FERRO-FAMIL
   Creation : 04/2005
   Update   :
*-------------------------------------------------------------------------------
   Description :  Selection of a SAR image Doppler sub-spectrum
*-------------------------------------------------------------------------------
   Inputs arguments :
   Returned values  :
   void
*******************************************************************************/
void select_sub_spectrum(float **fft_im,float **c_im,int offset,float *ham_win,int n_ham,float *vec1,int N,int Nrg)
{
 int az,rg;

 /*hamming window within the limits */
 for(az=0;az<N;az++) vec1[az] = 0;
 for(az=0;az<n_ham;az++) vec1[offset+az]=ham_win[az];
 for(rg=0;rg<Nrg;rg++)
 {
  for(az=0;az<N/2;az++)
  {
   c_im[rg][2*az]         = fft_im[rg][2*az]*vec1[az+N/2];
   c_im[rg][2*az+1]       = fft_im[rg][2*az+1]*vec1[az+N/2];
   c_im[rg][2*(az+N/2)]   = fft_im[rg][2*(az+N/2)]*vec1[az];
   c_im[rg][2*(az+N/2)+1] = fft_im[rg][2*(az+N/2)+1]*vec1[az];
  } 
  Fft(c_im[rg],N,-1);
 }
} 
 
