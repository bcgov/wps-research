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

File     : graphics.c
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
Description :  GRAPHICS Routines
*-------------------------------------------------------------------------------
Routines    :
void header(int nlig,int ncol,float Max,float Min,FILE *fbmp);
void header24(int nlig,int ncol,FILE *fbmp);
void headerTiff(int nlig,int ncol,FILE *fbmp);
void footerTiff(short int nlig, short int ncol, FILE * fptr);
void headerRas (int ncol,int nlig,float Max,float Min,FILE *fbmp);
void header24Ras (int ncol,int nlig,FILE *fbmp);
void colormap(int *red,int *green,int *blue,int comp);
void bmp_8bit(int nlig,int ncol,float Max,float Min,char *Colormap,float **DataBmp,char *name);
void bmp_8bit_char(int nlig,int ncol,float Max,float Min,char *Colormap,char *DataBmp,char *name);
void bmp_24bit(int nlig,int ncol,int mapgray,float **DataBmp,char *name);
void tiff_24bit(int nlig,int ncol,int mapgray,float **DataBmp,char *name);
void bmp_training_set(float **mat,int li,int co,char *nom,char *ColorMap16);
void bmp_wishart(float **mat,int li,int co,char *nom,char *ColorMap);
void bmp_h_alpha(float **mat, int li, int co, char *name, char *ColorMap);

*******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <dos.h>
#include <conio.h>
#endif

#include "rasterfile.h"
#include "graphics.h"
#include "matrix.h"
#include "processing.h"
#include "util.h"

/*******************************************************************************
Routine  : header
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates and writes a bitmap file header
*-------------------------------------------------------------------------------
Inputs arguments :
nlig     : BMP image number of lines
ncol     : BMP image number of rows
Max      : Coded Maximum Value
Min      : Coded Minimum Value
*fbmp    : BMP file pointer
Returned values  :
void
*******************************************************************************/
void header(int nlig, int ncol, float Max, float Min, FILE * fbmp)
{
    int k;
    int extracol;

/*Bitmap File Header*/
    k = 19778;
    fwrite(&k, sizeof(short int), 1, fbmp);
    extracol = (int) fmod(4 - (int) fmod(ncol, 4), 4);
    k = (ncol + extracol) * nlig + 1078;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 0;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 1078;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 40;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = ncol;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = nlig;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 1;
    fwrite(&k, sizeof(short int), 1, fbmp);
    k = 8;
    fwrite(&k, sizeof(short int), 1, fbmp);
    k = 0;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = ncol * nlig;
    fwrite(&k, sizeof(int), 1, fbmp);
    fwrite(&Max, sizeof(float), 1, fbmp);
    fwrite(&Min, sizeof(float), 1, fbmp);
    k = 256;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 256;
    fwrite(&k, sizeof(int), 1, fbmp);
}

/*******************************************************************************
Routine  : header24
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates and writes a 24 bit bitmap file header
*-------------------------------------------------------------------------------
Inputs arguments :
nlig   : BMP image number of lines
ncol   : BMP image number of rows
*fbmp  : BMP file pointer
Returned values  :
void
*******************************************************************************/
void header24(int nlig, int ncol, FILE * fbmp)
{
    int k;

/*Bitmap File Header*/
    k = 19778;
    fwrite(&k, sizeof(short int), 1, fbmp);
    k = (int) ((3 * ncol * nlig + 54) / 2);
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 0;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 54;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 40;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = ncol;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = nlig;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 1;
    fwrite(&k, sizeof(short int), 1, fbmp);
    k = 24;
    fwrite(&k, sizeof(short int), 1, fbmp);
    k = 0;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 3 * ncol * nlig;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 2952;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 2952;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 0;
    fwrite(&k, sizeof(int), 1, fbmp);
    k = 0;
    fwrite(&k, sizeof(int), 1, fbmp);
}

/*******************************************************************************
Routine  : headerTiff
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates and writes a 24 bit Tiff file header
*-------------------------------------------------------------------------------
Inputs arguments :
nlig   : TIFF image number of lines
ncol   : TIFF image number of rows
*fbmp  : TIFF file pointer
Returned values  :
void
*******************************************************************************/
void headerTiff(int nlig, int ncol, FILE * fptr)
{
   int offset;
   short int k = 18761;
   short int H42 = 42;

/*Tiff File Header*/
   /* Little endian & TIFF identifier */
   fwrite(&k, sizeof(short int), 1, fptr);
   fwrite(&H42, sizeof(short int), 1, fptr);

   offset = nlig * ncol * 3 + 8;
   fwrite(&offset, sizeof(int), 1, fptr);
   //putc((offset & 0xff000000) / 16777216,fptr);
   //putc((offset & 0x00ff0000) / 65536,fptr);
   //putc((offset & 0x0000ff00) / 256,fptr);
   //putc((offset & 0x000000ff),fptr);
}

/*******************************************************************************
Routine  : footerTiff
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates and writes a 24 bit Tiff file footer
*-------------------------------------------------------------------------------
Inputs arguments :
nlig   : TIFF image number of lines
ncol   : TIFF image number of rows
*fbmp  : TIFF file pointer
Returned values  :
void
*******************************************************************************/
void footerTiff(short int nlig, short int ncol, FILE * fptr)
{
   int offset;
   short int kk;
   short int H0 = 0;
   short int H1 = 1;
   short int H2 = 2;
   short int H3 = 3;
   short int H4 = 4;
   short int H5 = 5;
   short int H8 = 8;
   short int H14 = 14;
   short int H255 = 255;

/*Tiff File Footer*/
   /* The number of directory entries (14) */
   fwrite(&H14, sizeof(short int), 1, fptr);

   /* Width tag, short int */
   kk=256;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&ncol, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Height tag, short int */
   kk=257;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&nlig, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Bits per sample tag, short int */
   kk=258;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   offset = nlig * ncol * 3 + 182;
   fwrite(&offset, sizeof(int), 1, fptr);

   /* Compression flag, short int */
   kk=259;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Photometric interpolation tag, short int */
   kk=262;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H2, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Strip offset tag, long int */
   kk=273;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H4, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H8, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Orientation flag, short int */
   kk=274;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Sample per pixel tag, short int */
   kk=277;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Rows per strip tag, short int */
   kk=278;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&nlig, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Strip byte count flag, long int */
   kk=279;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H4, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   offset = nlig * ncol * 3;
   fwrite(&offset, sizeof(int), 1, fptr);

   /* X Resolution, short int */
   kk=282;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H5, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   offset = nlig * ncol * 3 + 188;
   fwrite(&offset, sizeof(int), 1, fptr);

   /* Y Resolution, short int */
   kk=283;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H5, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   offset = (int)nlig * (int)ncol * 3 + 196;
   fwrite(&offset, sizeof(int), 1, fptr);

   /* Planar configuration tag, short int */
   kk=284;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Sample format tag, short int */
   kk=296;fwrite(&kk, sizeof(short int), 1, fptr);
   fwrite(&H3, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H2, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* End of the directory entry */
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Bits for each colour channel */
   fwrite(&H8, sizeof(short int), 1, fptr);
   fwrite(&H8, sizeof(short int), 1, fptr);
   fwrite(&H8, sizeof(short int), 1, fptr);

/////////////////////////////////////////////////////
   /* Minimum value for each component */
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);
   fwrite(&H0, sizeof(short int), 1, fptr);

   /* Maximum value per channel */
   fwrite(&H255, sizeof(short int), 1, fptr);
   fwrite(&H255, sizeof(short int), 1, fptr);
   fwrite(&H255, sizeof(short int), 1, fptr);

   /* Samples per pixel for each channel */
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
   fwrite(&H1, sizeof(short int), 1, fptr);
}

/*******************************************************************************
Routine  : headerRas
Authors  : Stephane MERIC
Creation : 12/2006
Update   :
*-------------------------------------------------------------------------------
Description :  Creates and writes a SUN raster file header
*-------------------------------------------------------------------------------
Inputs arguments :
ncol     : SUN raster image number of rows
nlig     : SUN raster image number of lines
Max      : Coded Maximum Value
Min      : Coded Minimum Value
*fbmp    : BMP file pointer
Returned values  :

*******************************************************************************/
void headerRas(int ncol, int nlig, float Max, float Min, FILE * fbmp)
{
    int k;

/* writing Max and Min values at the end of the file fbmp */    
    fseek(fbmp, 32 + 768 + (nlig * ncol), SEEK_SET);
    fwrite(&Max, sizeof(float), 1, fbmp);
    fwrite(&Min, sizeof(float), 1, fbmp);
    rewind(fbmp);

/*Sun raster File Header*/
    k = RAS_MAGIC; /* 0x59a66a95 : magic number for SUN raster file */
    fwrite(&k, sizeof(int), 1, fbmp);

    k = ncol; /* ras_width : width (pixels) of image */
    fwrite(&k, sizeof(int), 1, fbmp);

    k = nlig; /* ras_height : height (pixels) of image */
    fwrite(&k, sizeof(int), 1, fbmp);

    k = 8; /* ras_depth : depth (1, 8, 24, 32 bits) of pixel */
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = nlig * ncol; /* ras_length : length (bytes) of image */
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = RT_STANDARD; /* ras_type : type of file -> raw pixrect image in 68000 byte order */
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = RMT_EQUAL_RGB; /* ras_maptype : type of colormap -> no colormap */ 
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = 768; /* ras_maplength : length (bytes) of following map */
    fwrite(&k, sizeof(int), 1, fbmp);
    
}

/*******************************************************************************
Routine  : header24Ras
Authors  : Stephane MERIC
Creation : 12/2006
Update   :
*-------------------------------------------------------------------------------
Description :  Creates and writes a 24 bit SUN raster file header
*-------------------------------------------------------------------------------
Inputs arguments :
ncol     : SUN raster image number of rows
nlig     : SUN raster image number of lines
*fbmp    : BMP file pointer
Returned values  :

*******************************************************************************/
void header24Ras(int ncol, int nlig, FILE * fbmp)
{
    int k;

/*Sun raster File Header*/
    k = RAS_MAGIC; /* 0x59a66a95 : magic number for SUN raster file */
    fwrite(&k, sizeof(int), 1, fbmp);

    k = ncol; /* ras_width : width (pixels) of image */
    fwrite(&k, sizeof(int), 1, fbmp);

    k = nlig; /* ras_height : height (pixels) of image */
    fwrite(&k, sizeof(int), 1, fbmp);

    k = 24; /* ras_depth : depth (1, 8, 24, 32 bits) of pixel */
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = ncol * nlig * 24 / 8; /* ras_length : length (bytes) of image */
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = RT_STANDARD; /* ras_type : type of file -> raw pixrect image in 68000 byte order */
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = RMT_NONE; /* ras_maptype : type of colormap -> red, green, blue */ 
    fwrite(&k, sizeof(int), 1, fbmp);
    
    k = 0; /* ras_maplength : length (bytes) of following map */
    fwrite(&k, sizeof(int), 1, fbmp);
}

/*******************************************************************************
Routine  : colormap
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates a jet, hsv or gray 256 element colormap
*-------------------------------------------------------------------------------
Inputs arguments :
red     : red channel vector
green   : green channel vector
blue    : blue channel vector
comp    : colormap selector
Returned values  :
all color channels
*******************************************************************************/
void colormap(int *red, int *green, int *blue, int comp)
{
    int k;

    if (comp == 1) {
/**********************************************************************/
/*Definition of the Gray(256) Colormap*/
	for (k = 0; k < 256; k++) {
	    red[k] = k;
	    green[k] = k;
	    blue[k] = k;
	}
    }


    if (comp == 2) {
/**********************************************************************/
/*Definition of the HSV(256) Colormap*/
	for (k = 0; k < 43; k++) {
	    blue[k] = 255;
	    green[k] = (int) (k * 0.0234 * 255);
	    red[k] = 0;
	}
	for (k = 0; k < 43; k++) {
	    blue[43 + k] = (int) (255 * (0.9922 - k * 0.0234));
	    green[43 + k] = 255;
	    red[43 + k] = 0;
	}
	for (k = 0; k < 42; k++) {
	    blue[86 + k] = 0;
	    green[86 + k] = 255;
	    red[86 + k] = (int) (255 * (0.0156 + k * 0.0234));
	}
	for (k = 0; k < 43; k++) {
	    blue[128 + k] = 0;
	    green[128 + k] = (int) (255 * (1. - k * 0.0234));
	    red[128 + k] = 255;
	}
	for (k = 0; k < 43; k++) {
	    blue[171 + k] = (int) (255 * (0.0078 + k * 0.0234));
	    green[171 + k] = 0;
	    red[171 + k] = 255;
	}
	for (k = 0; k < 42; k++) {
	    blue[214 + k] = 255;
	    green[214 + k] = 0;
	    red[214 + k] = (int) (255 * (0.9844 - k * 0.0234));
	}
    }


    if (comp == 3) {
/**********************************************************************/
/*Definition of the Jet(256) Colormap*/
	for (k = 0; k < 32; k++) {
	    red[k] = 128 + 4 * k;
	    green[k] = 0.;
	    blue[k] = 0.;
	}
	for (k = 0; k < 64; k++) {
	    red[32 + k] = 255;
	    green[32 + k] = 4 * k;
	    blue[32 + k] = 0.;
	}
	for (k = 0; k < 64; k++) {
	    red[96 + k] = 252 - 4 * k;
	    green[96 + k] = 255;
	    blue[96 + k] = 4 * k;
	}
	for (k = 0; k < 64; k++) {
	    red[160 + k] = 0;
	    green[160 + k] = 252 - 4 * k;
	    blue[160 + k] = 255;
	}
	for (k = 0; k < 32; k++) {
	    red[224 + k] = 0;
	    green[224 + k] = 0.;
	    blue[224 + k] = 252 - 4 * k;
	}
    }
}

/*******************************************************************************
Routine  : bmp_8bit
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   : 12/2006 (Stephane MERIC)
*-------------------------------------------------------------------------------
Description :  Creates a 8 bit BMP file
*-------------------------------------------------------------------------------
Inputs arguments :
nlig      : matrix number of lines
ncol      : matrix number of rows
Max       : Maximum value
Min       : Minimum value
*ColorMap : ColorMap name
**mat     : matrix containg float values
*name     : BMP file name (without the .bmp extension)
Returned values  :
void
*******************************************************************************/
void
bmp_8bit(int nlig, int ncol, float Max, float Min, char *ColorMap,
	 float **DataBmp, char *name)
{
    FILE *fbmp;
    FILE *fcolormap;

    char *bufimg;
    char *bufcolor;
    char Tmp[1024];

    int lig, col, k, l;
    int ncolbmp, extracol, Ncolor;
    int red[256], green[256], blue[256];

    extracol = (int) fmod(4 - (int) fmod(ncol, 4), 4);
    ncolbmp = ncol + extracol;

    bufimg = vector_char(nlig * ncolbmp);
    bufcolor = vector_char(1024);

    if ((fbmp = fopen(name, "wb")) == NULL)
	edit_error("ERREUR DANS L'OUVERTURE DU FICHIER", name);
/*****************************************************************************/
/* Definition of the Header */

    #if defined(__sun) || defined(__sun__)
    	headerRas(ncol, nlig, Max, Min, fbmp);
    #else
    	header(nlig, ncol, Max, Min, fbmp);
    #endif

/*****************************************************************************/
/* Definition of the Colormap */
    if ((fcolormap = fopen(ColorMap, "r")) == NULL)
       edit_error("Could not open the bitmap file ",ColorMap);
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%i\n", &Ncolor);
    for (k = 0; k < Ncolor; k++)
	fscanf(fcolormap, "%i %i %i\n", &red[k], &green[k], &blue[k]);
    fclose(fcolormap);

/* Bitmap colormap and BMP writing */
    for (col = 0; col < 1024; col++)
	bufcolor[col] = (char) (0);

    #if defined(__sun) || defined(__sun__)

	for (col = 0; col < Ncolor; col++) {

		bufcolor[col] = (char) (red[col]);
		bufcolor[col + 256] = (char) (green[col]);
		bufcolor[col + 512] = (char) (blue[col]);
		}
	
	fwrite(&bufcolor[0], sizeof(char), 768, fbmp);

	for (lig = 0; lig < nlig; lig++) {
	
	    	for (col = 0; col < ncol; col++) {
	    		l = (int) DataBmp[lig][col];
	    		bufimg[lig * ncolbmp + col] = (char) l;
			}
		
	    	for (col = 0; col < extracol; col++) {
	    		l = 0;
	    		bufimg[lig * ncolbmp + col] = (char) l;
			}
	}
   
    #else

	for (col = 0; col < Ncolor; col++) {	
		
		bufcolor[4 * col] = (char) (blue[col]);
		bufcolor[4 * col + 1] = (char) (green[col]);
		bufcolor[4 * col + 2] = (char) (red[col]);
		bufcolor[4 * col + 3] = (char) (0);
	}
		
	fwrite(&bufcolor[0], sizeof(char), 1024, fbmp);
	
	for (lig = 0; lig < nlig; lig++) {
	    
	    	for (col = 0; col < ncol; col++) {
	    		l = (int) DataBmp[nlig - lig - 1][col];
	    		bufimg[lig * ncolbmp + col] = (char) l;
			}
	    	
	    	for (col = 0; col < extracol; col++) {
	    		l = 0;
	    		bufimg[lig * ncolbmp + col] = (char) l;
			}
	}
   			
    #endif

    fwrite(&bufimg[0], sizeof(char), nlig * ncolbmp, fbmp);

    free_vector_char(bufcolor);
    free_vector_char(bufimg);
    fclose(fbmp);
}

/*******************************************************************************
Routine  : bmp_8bit_char
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  ReCreates a 8 bit BMP file
*-------------------------------------------------------------------------------
Inputs arguments :
nlig      : matrix number of lines
ncol      : matrix number of rows
Max       : Maximum value
Min       : Minimum value
*ColorMap : ColorMap name
*mat     : vector containg char values
*name     : BMP file name (without the .bmp extension)
Returned values  :
void
*******************************************************************************/
void bmp_8bit_char(int nlig, int ncol, float Max, float Min, char *ColorMap, char *DataBmp, char *name)
{
    FILE *fbmp;
    FILE *fcolormap;

    char *bufimg;
    char *bufcolor;
    char Tmp[1024];

    int lig, col, k, l;
    int ncolbmp, extracol, Ncolor;
    int red[256], green[256], blue[256];

    extracol = (int) fmod(4 - (int) fmod(ncol, 4), 4);
    ncolbmp = ncol + extracol;

    bufimg = vector_char(nlig * ncolbmp);
    bufcolor = vector_char(1024);

    if ((fbmp = fopen(name, "wb")) == NULL)
	edit_error("ERREUR DANS L'OUVERTURE DU FICHIER", name);
/*****************************************************************************/
/* Definition of the Header */

    #if defined(__sun) || defined(__sun__)
    	headerRas(ncol, nlig, Max, Min, fbmp);
    #else
    	header(nlig, ncol, Max, Min, fbmp);
    #endif

/*****************************************************************************/
/* Definition of the Colormap */
    if ((fcolormap = fopen(ColorMap, "r")) == NULL)
       edit_error("Could not open the bitmap file ",ColorMap);
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%i\n", &Ncolor);
    for (k = 0; k < Ncolor; k++)
	fscanf(fcolormap, "%i %i %i\n", &red[k], &green[k], &blue[k]);
    fclose(fcolormap);

/* Bitmap colormap and BMP writing */
    for (col = 0; col < 1024; col++)
	bufcolor[col] = (char) (0);

    #if defined(__sun) || defined(__sun__)

	for (col = 0; col < Ncolor; col++) {

		bufcolor[col] = (char) (red[col]);
		bufcolor[col + 256] = (char) (green[col]);
		bufcolor[col + 512] = (char) (blue[col]);
		}
	
	fwrite(&bufcolor[0], sizeof(char), 768, fbmp);

    #else

	for (col = 0; col < Ncolor; col++) {	
		
		bufcolor[4 * col] = (char) (blue[col]);
		bufcolor[4 * col + 1] = (char) (green[col]);
		bufcolor[4 * col + 2] = (char) (red[col]);
		bufcolor[4 * col + 3] = (char) (0);
		}
		
	fwrite(&bufcolor[0], sizeof(char), 1024, fbmp);
	   			
    #endif

/*****************************************************************************/
    for (lig = 0; lig < nlig; lig++) {
	for (col = 0; col < ncol; col++) {
	    bufimg[lig * ncolbmp + col] = DataBmp[lig * ncol + col];
	}
	for (col = 0; col < extracol; col++) {
	    l = 0;
	    bufimg[lig * ncolbmp + col] = (char) l;
	}
    }
    fwrite(&bufimg[0], sizeof(char), nlig * ncolbmp, fbmp);

    free_vector_char(bufcolor);
    free_vector_char(bufimg);
    fclose(fbmp);
}

/*******************************************************************************
Routine  : bmp_24bit
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   : 12/2006 (Stephane MERIC)
*-------------------------------------------------------------------------------
Description :  Creates a 24 bit BMP file
*-------------------------------------------------------------------------------
Inputs arguments :
nlig      : matrix number of lines
ncol      : matrix number of rows
mapgray   : ColorMap Gray or not (0/1)
**mat     : matrix containg float values
*name     : BMP file name (without the .bmp extension)
Returned values  :
void
*******************************************************************************/
void
bmp_24bit(int nlig, int ncol, int mapgray, float **DataBmp, char *name)
{
    FILE *fbmp;

    char *bmpimg;

    int lig, col, l;
    int ncolbmp;

    float hue, red, green, blue;
    float m1, m2, h;

    ncolbmp = ncol - (int) fmod((float) ncol, 4.);
    bmpimg = vector_char(3 * nlig * ncolbmp);


   if ((fbmp = fopen(name, "wb")) == NULL)
	edit_error("ERREUR DANS L'OUVERTURE DU FICHIER", name);
/*****************************************************************************/
/* Definition of the Header */

    #if defined(__sun) || defined(__sun__)
    	header24Ras(ncol, nlig, fbmp);
    #else
    	header24(nlig, ncol, fbmp);
    #endif

/*****************************************************************************/
// CONVERSION HSV TO RGB with V=0.5 ans S=1
    m2 = 1.;
    m1 = 0.;
    for (lig = 0; lig < nlig; lig++) {
	for (col = 0; col < ncolbmp; col++) {
	    hue = DataBmp[lig][col];

        if (mapgray == 1) {
		red = hue/360.;
		green = hue/360.;
		blue = hue/360.;
	    } else {
		h = hue + 120;
		if (h > 360.) h = h - 360.;
		else if (h < 0.) h = h + 360.;
		if (h < 60.) red = m1 + (m2 - m1) * h / 60.;
		else if (h < 180.) red = m2;
		else if (h < 240.) red = m1 + (m2 - m1) * (240. - h) / 60.;
		else red = m1;

		h = hue;
		if (h > 360.) h = h - 360.;
		else if (h < 0.) h = h + 360.;
		if (h < 60.) green = m1 + (m2 - m1) * h / 60.;
		else if (h < 180.) green = m2;
		else if (h < 240.) green = m1 + (m2 - m1) * (240. - h) / 60.;
		else green = m1;

		h = hue - 120;
		if (h > 360.) h = h - 360.;
		else if (h < 0.) h = h + 360.;
		if (h < 60.) blue = m1 + (m2 - m1) * h / 60.;
		else if (h < 180.) blue = m2;
		else if (h < 240.) blue = m1 + (m2 - m1) * (240. - h) / 60.;
		else blue = m1;

	    }
	    #if defined(__sun__) || defined(__sun)

	    	if (blue > 1.) blue = 1.;
	    	if (blue < 0.) blue = 0.;
	    	l = (int) (floor(255 * blue));
	    	bmpimg[3 * (lig) * ncolbmp + 3 * col + 0] = (char) (l);
	    	if (green > 1.) green = 1.;
	    	if (green < 0.) green = 0.;
	    	l = (int) (floor(255 * green));
	    	bmpimg[3 * (lig) * ncolbmp + 3 * col + 1] =	(char) (l);
	    	if (red > 1.) red = 1.;
	    	if (red < 0.) red = 0.;
	    	l = (int) (floor(255 * red));
	    	bmpimg[3 * (lig) * ncolbmp + 3 * col + 2] =	(char) (l);
  	    
  	    #else
	    	
	    	if (blue > 1.) blue = 1.;
	    	if (blue < 0.) blue = 0.;
	    	l = (int) (floor(255 * blue));
	    	bmpimg[3 * (nlig - lig - 1) * ncolbmp + 3 * col + 0] = (char) (l);
	    	if (green > 1.) green = 1.;
	    	if (green < 0.) green = 0.;
	    	l = (int) (floor(255 * green));
	    	bmpimg[3 * (nlig - lig - 1) * ncolbmp + 3 * col + 1] =	(char) (l);
	    	if (red > 1.) red = 1.;
	    	if (red < 0.) red = 0.;
	    	l = (int) (floor(255 * red));
	    	bmpimg[3 * (nlig - lig - 1) * ncolbmp + 3 * col + 2] =	(char) (l);
  	   
  	   #endif
  		  	     
     }
     }


    fwrite(&bmpimg[0], sizeof(char), 3 * nlig * ncolbmp, fbmp);

    free_vector_char(bmpimg);
    fclose(fbmp);
}

/*******************************************************************************
Routine  : tiff_24bit
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates a 24 bit TIFF file
*-------------------------------------------------------------------------------
Inputs arguments :
nlig      : matrix number of lines
ncol      : matrix number of rows
mapgray   : ColorMap Gray or not (0/1)
**mat     : matrix containg float values
*name     : TIFF file name (without the .bmp extension)
Returned values  :
void
*******************************************************************************/
void
tiff_24bit(int nlig, int ncol, int mapgray, float **DataBmp, char *name)
{
    FILE *fptr;

    char *bmpimg;

    int lig, col, l;
    int ncolbmp;

    float hue, red, green, blue;
    float m1, m2, h;

    ncolbmp = ncol - (int) fmod((float) ncol, 4.);
    bmpimg = vector_char(3 * nlig * ncolbmp);

    if ((fptr = fopen(name, "wb")) == NULL)
	edit_error("ERREUR DANS L'OUVERTURE DU FICHIER", name);
/*****************************************************************************/
/* Definition of the Header */

    headerTiff(nlig, ncolbmp, fptr);

/*****************************************************************************/
// CONVERSION HSV TO RGB with V=0.5 ans S=1
    m2 = 1.;
    m1 = 0.;
    for (lig = 0; lig < nlig; lig++) {
	for (col = 0; col < ncolbmp; col++) {
	    hue = DataBmp[lig][col];

        if (mapgray == 1) {
		red = hue/360.;
		green = hue/360.;
		blue = hue/360.;
	    } else {
		h = hue + 120;
		if (h > 360.) h = h - 360.;
		else if (h < 0.) h = h + 360.;
		if (h < 60.) red = m1 + (m2 - m1) * h / 60.;
		else if (h < 180.) red = m2;
		else if (h < 240.) red = m1 + (m2 - m1) * (240. - h) / 60.;
		else red = m1;

		h = hue;
		if (h > 360.) h = h - 360.;
		else if (h < 0.) h = h + 360.;
		if (h < 60.) green = m1 + (m2 - m1) * h / 60.;
		else if (h < 180.) green = m2;
		else if (h < 240.) green = m1 + (m2 - m1) * (240. - h) / 60.;
		else green = m1;

		h = hue - 120;
		if (h > 360.) h = h - 360.;
		else if (h < 0.) h = h + 360.;
		if (h < 60.) blue = m1 + (m2 - m1) * h / 60.;
		else if (h < 180.) blue = m2;
		else if (h < 240.) blue = m1 + (m2 - m1) * (240. - h) / 60.;
		else blue = m1;

	    }

	    if (blue > 1.) blue = 1.;
	    if (blue < 0.) blue = 0.;
	    l = (int) (floor(255 * blue));
	    bmpimg[3 * (lig) * ncolbmp + 3 * col + 2] = (char) (l);
	    if (green > 1.) green = 1.;
	    if (green < 0.) green = 0.;
	    l = (int) (floor(255 * green));
	    bmpimg[3 * (lig) * ncolbmp + 3 * col + 1] =	(char) (l);
	    if (red > 1.) red = 1.;
	    if (red < 0.) red = 0.;
	    l = (int) (floor(255 * red));
	    bmpimg[3 * (lig) * ncolbmp + 3 * col + 0] =	(char) (l);
     }
     }

    fwrite(&bmpimg[0], sizeof(char), 3 * nlig * ncolbmp, fptr);

/*****************************************************************************/
/* Definition of the Footer */

    footerTiff(nlig, ncolbmp, fptr);

/*****************************************************************************/

    free_vector_char(bmpimg);
    fclose(fptr);
}

/*******************************************************************************
Routine  : bmp_training_set
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates a bitmap file of the training areas
*-------------------------------------------------------------------------------
Inputs arguments :
mat   : matrix containg float values
nlig  : matrix number of lines
ncol  : matrixnumber of rows
*name : BMP file name (without the .bmp extension)
Returned values  :
void
*******************************************************************************/
void
bmp_training_set(float **mat, int li, int co, char *nom, char *ColorMap16)
{
    FILE *fbmp;
    FILE *fcolormap;

    char *bufimg;
    char *bufcolor;
    char Tmp[1024];

    float min, max;
    int lig, col, k, l, nlig, ncol, Ncolor;
    int red[256], green[256], blue[256];

    nlig = li;
    ncol = co - (int) fmod((double) co, (double) 4);
    bufimg = vector_char(nlig * ncol);
    bufcolor = vector_char(1024);

    strcat(nom, ".bmp");
    if ((fbmp = fopen(nom, "wb")) == NULL)
	edit_error("ERREUR DANS L'OUVERTURE DU FICHIER", nom);

    min = 1;
    max = -20;
    for (lig = 0; lig < nlig; lig++)
	for (col = 0; col < ncol; col++)
	    if (mat[lig][col] > max)
		max = mat[lig][col];

    #if defined(__sun) || defined(__sun__)
    	headerRas(ncol, nlig, max, min, fbmp);
    #else
    	header(nlig, ncol, max, min, fbmp);
    #endif


/*                 Definition of the Colormap                         */
    if ((fcolormap = fopen(ColorMap16, "r")) == NULL)
	edit_error("Could not open the bitmap file ", ColorMap16);

/* Colormap Definition  */
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%i\n", &Ncolor);
    for (k = 0; k < Ncolor; k++)
	fscanf(fcolormap, "%i %i %i\n", &red[k], &green[k], &blue[k]);
    fclose(fcolormap);

/* Bitmap colormap writing */
    #if defined(__sun) || defined(__sun__)
    
    	for (k = 0; k < Ncolor; k++) {
		
		bufcolor[k] = (char) (1);
		bufcolor[k + 256] = (char) (1);
		bufcolor[k + 512] = (char) (1);
  		}
  		
	for (k = 0; k <= floor(max); k++) {
	
		bufcolor[k] = (char) (red[k]);
		bufcolor[k + 256] = (char) (green[k]);
		bufcolor[k + 512] = (char) (blue[k]);
		}
		
	fwrite(&bufcolor[0], sizeof(char), 768, fbmp);

	/* Image writing */
    	for (lig = 0; lig < nlig; lig++) {
		for (col = 0; col < ncol; col++) {
	    		l = (int) mat[lig][col];
	    		bufimg[lig * ncol + col] = (char) l;
			}
    		}
    	
    	fwrite(&bufimg[0], sizeof(char), nlig * ncol, fbmp);

    #else

	/* Bitmap colormap writing */    
    	for (k = 0; k < Ncolor; k++) {
		
		bufcolor[4 * k] = (char) (1);
		bufcolor[4 * k + 1] = (char) (1);
		bufcolor[4 * k + 2] = (char) (1);
		bufcolor[4 * k + 3] = (char) (0);
		}   	

    	for (k = 0; k <= floor(max); k++) {

		bufcolor[4 * k] = (char) (blue[k]);
		bufcolor[4 * k + 1] = (char) (green[k]);
		bufcolor[4 * k + 2] = (char) (red[k]);
		bufcolor[4 * k + 3] = (char) (0);
		}
		
	fwrite(&bufcolor[0], sizeof(char), 1024, fbmp);

	/* Image writing */
    	for (lig = 0; lig < nlig; lig++) {
		for (col = 0; col < ncol; col++) {
	    		l = (int) mat[nlig - lig - 1][col];
	    		bufimg[lig * ncol + col] = (char) l;
			}
    		}
    	
    	fwrite(&bufimg[0], sizeof(char), nlig * ncol, fbmp);
	
    #endif
     	
    free_vector_char(bufcolor);
    free_vector_char(bufimg);
    fclose(fbmp);
}

/*******************************************************************************
Routine  : bmp_wishart
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates a bitmap file from a matrix resulting from the wishart
H / A / Alpha classification
*-------------------------------------------------------------------------------
Inputs arguments :
mat   : matrix containg float values
nlig  : matrix number of lines
ncol  : matrixnumber of rows
*name : BMP file name (without the .bmp extension)
Returned values  :
void
*******************************************************************************/
void bmp_wishart(float **mat, int li, int co, char *nom, char *ColorMap)
{
    FILE *fbmp;
    FILE *fcolormap;

    char *bufimg;
    char *bufcolor;
    char Tmp[10];

    float min, max;
    int lig, col, k, l, nlig, ncol, Ncolor;
    int red[256], green[256], blue[256];

    nlig = li;
    ncol = co - (int) fmod((double) co, (double) 4);
    bufimg = vector_char(nlig * ncol);
    bufcolor = vector_char(1024);

    strcat(nom, ".bmp");
    if ((fbmp = fopen(nom, "wb")) == NULL)
	edit_error("ERREUR DANS L'OUVERTURE DU FICHIER", nom);

    min = 1;
    max = -20;
    for (lig = 0; lig < nlig; lig++)
	for (col = 0; col < ncol; col++)
	    if (mat[lig][col] > max)
		max = mat[lig][col];

    #if defined(__sun) || defined(__sun__)
    	headerRas(ncol, nlig, max, min, fbmp);
    #else
    	header(nlig, ncol, max, min, fbmp);
    #endif

/*                 Definition of the Colormap                         */
    if ((fcolormap = fopen(ColorMap, "r")) == NULL)
	edit_error("Could not open the bitmap file ", ColorMap);

/* Colormap Definition  */
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%i\n", &Ncolor);
    for (k = 0; k < Ncolor; k++)
	fscanf(fcolormap, "%i %i %i\n", &red[k], &green[k], &blue[k]);
    fclose(fcolormap);

    #if defined(__sun) || defined(__sun__)
    
	for (col = 0; col < 256; col++) {
	
		bufcolor[col] = (char) (red[col]);
		bufcolor[col + 256] = (char) (green[col]);
		bufcolor[col + 512] = (char) (blue[col]);
		}
		
	fwrite(&bufcolor[0], sizeof(char), 768, fbmp);

	/* Image writing */
    	for (lig = 0; lig < nlig; lig++) {
		
		for (col = 0; col < ncol; col++) {
	    		l = (int) mat[lig][col];
	    		bufimg[lig * ncol + col] = (char) l;
			}
    		}
    	
    	fwrite(&bufimg[0], sizeof(char), nlig * ncol, fbmp);

    #else

	/* Bitmap colormap writing */    
    	for (col = 0; col < 256; col++) {
		
		bufcolor[4 * col] = (char) (blue[col]);
		bufcolor[4 * col + 1] = (char) (green[col]);
		bufcolor[4 * col + 2] = (char) (red[col]);
		bufcolor[4 * col + 3] = (char) (0);
		}
		
	fwrite(&bufcolor[0], sizeof(char), 1024, fbmp);

	/* Image writing */
    	for (lig = 0; lig < nlig; lig++) {
		
		for (col = 0; col < ncol; col++) {
	    		l = (int) mat[nlig - lig - 1][col];
	    		bufimg[lig * ncol + col] = (char) l;
			}
    		}
    	
    	fwrite(&bufimg[0], sizeof(char), nlig * ncol, fbmp);
	
    #endif


    free_vector_char(bufcolor);
    free_vector_char(bufimg);
    fclose(fbmp);
}

/*******************************************************************************
Routine  : bmp_h_alpha
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Creation : 01/2002
Update   :
*-------------------------------------------------------------------------------
Description :  Creates a bitmap file from a matrix resulting from the H-Alpha classification
*-------------------------------------------------------------------------------
Inputs arguments :
mat   : matrix containg float values ranging from 1 to 9
nlig  : matrix number of lines
ncol  : matrixnumber of rows
*name : BMP file name (without the .bmp extension)
Returned values  :
void
*******************************************************************************/
void bmp_h_alpha(float **mat, int li, int co, char *name, char *ColorMap)
{
    FILE *fbmp;
    FILE *fcolormap;

    char *bufimg;
    char *bufcolor;
    char Tmp[1024];

    int lig, col, k, l, nlig, ncol, Ncolor;
    int red[256], green[256], blue[256];

    float MinBMP, MaxBMP;

    nlig = li;
    ncol = co - (int) fmod((double) co, (double) 4);	/* The number of rows has tobe a factor of 4 */
    bufimg = vector_char(nlig * ncol);
    bufcolor = vector_char(1024);

/* Bitmap file opening */
    strcat(name, ".bmp");
    if ((fbmp = fopen(name, "wb")) == NULL)
	edit_error("Could not open the bitmap file ", name);

/* Bitmap header writing */
    MinBMP = 1.;
    MaxBMP = 9.;
    #if defined(__sun) || defined(__sun__)
    	headerRas(ncol, nlig, MaxBMP, MinBMP, fbmp);
    #else
    	header(nlig, ncol, MaxBMP, MinBMP, fbmp);
    #endif

/* Colormap Definition  1 to 9*/
    if ((fcolormap = fopen(ColorMap, "r")) == NULL)
	edit_error("Could not open the file ", ColorMap);

/* Colormap Definition  */
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%s\n", Tmp);
    fscanf(fcolormap, "%i\n", &Ncolor);
    for (k = 0; k < Ncolor; k++)
	fscanf(fcolormap, "%i %i %i\n", &red[k], &green[k], &blue[k]);
    fclose(fcolormap);

    #if defined(__sun) || defined(__sun__)
    
	for (col = 0; col < 256; col++) {
	
		bufcolor[col] = (char) (red[col]);
		bufcolor[col + 256] = (char) (green[col]);
		bufcolor[col + 512] = (char) (blue[col]);
		}
		
	fwrite(&bufcolor[0], sizeof(char), 768, fbmp);

	/* Image writing */
    	for (lig = 0; lig < nlig; lig++) {
		
		for (col = 0; col < ncol; col++) {
	    		l = (int) mat[lig][col];
	    		bufimg[lig * ncol + col] = (char) l;
			}
    		}
    	
    #else

	/* Bitmap colormap writing */    
    	for (col = 0; col < 256; col++) {
		
		bufcolor[4 * col] = (char) (blue[col]);
		bufcolor[4 * col + 1] = (char) (green[col]);
		bufcolor[4 * col + 2] = (char) (red[col]);
		bufcolor[4 * col + 3] = (char) (0);
		}
		
	fwrite(&bufcolor[0], sizeof(char), 1024, fbmp);

	/* Image writing */
    	for (lig = 0; lig < nlig; lig++) {
		
		for (col = 0; col < ncol; col++) {
	    		l = (int) mat[nlig - lig - 1][col];
	    		bufimg[lig * ncol + col] = (char) l;
			}
    		}
    	
    #endif

    fwrite(&bufimg[0], sizeof(char), nlig * ncol, fbmp);

    free_vector_char(bufcolor);
    free_vector_char(bufimg);
    fclose(fbmp);
}
