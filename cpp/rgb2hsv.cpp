#include<math.h>
#include"misc.h"
/* from
https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
 

Need to look at:
https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
also
*/
typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;     
}
int main(int argc, char ** argv){
  if(argc < 2) err("rgb2hsv.exe [input raster file name, 3 band]");

  str fn(argv[1]); // input image file name
  if(!exists(fn)) err(str("failed to open input file: ") + fn);
  str hfn(hdr_fn(fn)); // input header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  if(nband != 3) err("3 band input supported"); // need rgb

  size_t i, j, k;
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(nrow * ncol * nband);

  float h, s, v, r, g, b;
  for0(i, nrow){
    for0(j, ncol){
      k = i * ncol + j;
      r = dat[k];
      g = dat[k + np];
      b = dat[k + np + np];

      rgb x;
      x.r = r;
      x.g = g;
      x.b = b;

     hsv y = rgb2hsv(x); //&r, &g, &b, h, s, v);
      out[k] = y.h;
      out[k + np] = y.s;
      out[k + np + np] = y.v;
    }
  }


  str ofn(fn + str("_hsv.bin"));
  str ohn(fn + str("_hsv.hdr"));

  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
 
  return 0;
}

