/* matrix2.h by Ashlin Richardson, Senior Data Scientist, BC Wildfire Service */
#pragma once
#include<cmath>
#include<math.h>
#include<string>
#include<fstream>
#include<iostream>

#define TYPE double
#define _angle(x) (atan2(imag(x), real(x)))
#define cp(c) printf("%f + %fi\n", real(c), imag(c));

using namespace std;

template<class T> struct complex{
  T re; T im;

  inline complex<T>(): re(0.), im(0.){
  }

  inline complex( T Re, T Im) : re(Re), im(Im){
  }

  inline complex(T scalar): re(scalar), im(0.){
  }

  inline void zero(){
    re = 0.;
    im = 0.;
  }

  inline void real(T dat){
    re = dat;
  }

  inline void imag(T dat){
    im = dat;
  }

  inline void operator = (const complex &A){
    re = A.re;
    im = A.im;
  }

  inline void operator = (const T &A){
    re = A; // added 20210602
    im = 0;
  }

  inline complex<T>(const complex<double> &A): re(A.re), im(A.im){
  }

  inline complex<T>(const complex<float> &A): re((double)A.re), im((double)A.im){
  }

  inline complex<T> log(){
    return complex<T>((T)(double)std::log((double)sqrt((double)(re*re + im*im))),
                      (T)(double)atan2((double)im, (double)re));
  }
  inline double angle(){
    return atan2(im, re); //, real(*this));
  }
};

template<class T> inline ostream & operator<<( ostream &output, const complex<T> A){
  output.precision(5);
  output.fill(' ');
  output << right ;
  output << scientific;
  if( A.im < 0){
    output << A.re << A.im << "i" ;
  }
  else{
    output << A.re << "+" << A.im << "i" ;
  }
  return output;
}

template<class T> inline complex<T> operator+(const complex<T> &A, const complex<T> &B){
  complex<T> ret(B.re + A.re, B.im + A.im);
  return ret;
}

template<class T> inline complex<T> operator-(const complex<T> &A, const complex<T> &B){
  complex<T> ret(A.re - B.re, A.im - B.im);
  return ret;
}

template<class T> inline complex<T> operator-(const complex<T> &B){
  complex<T> ret(-B.re, -B.im);
  return ret;
}

template<class T> inline complex<T> operator*(const complex<T> &A, const complex<T> &B){
  complex<T> ret(((A.re)*(B.re))-((A.im)*(B.im)) , ((A.im)*(B.re))+((A.re)*(B.im)) );
  return ret;
}

template<class T> inline complex<T> conj(complex<T> const &A){
  complex<T> ret( A.re, - A.im );
  return ret;
}

template<class T> inline T real(complex<T> const &A){
  return A.re;
}

template<class T> inline T imag(complex<T> const &A){
  return A.im;
}

template<class T> inline T abs(complex<T> const &A){
  return sqrt((T)((A.re)*(A.re)) + ((A.im)*(A.im)));
}

template<class T> inline complex<T> operator/(const complex<T> &A, const complex<T> &B){
  T babs = abs(B);
  babs = babs*babs;
  complex<T> ret( (((A.re)*(B.re))+((A.im)*(B.im)))/babs ,
  (((A.im)*(B.re))-((A.re)*(B.im)))/babs );
  return ret;
}

template<class T> inline void operator/=(complex<T> &A, const T &B){
  A.re /= B; /* inplace right scalar divide */
  A.im /= B;
}

template<class T> inline T arg(complex<T> const &A){
  return atan2(A.im , A.re);
}

template<class T> inline complex<T> polar(T r, T theta){
  complex<T> ret(r*cos(theta), r*sin(theta)); // initialize cplx from polar
  return ret;
}

template<class T> inline complex<T> sqrt(complex<T> const &A){
  return polar( sqrt(abs(A)), arg(A)/2.);
}

template<class T> inline complex<T> pow(complex<T> const &A, TYPE exponent){
  T r = abs(A);
  T theta = arg(A);
  T new_r = pow(r, exponent);
  T new_theta = exponent*theta;
  return( complex <T> (new_r*cos(new_theta),new_r*sin(new_theta)));
}

template<class T> inline complex<T> exp(complex<T> const &A){
  T e = std::exp(real(A));
  return complex<T> (e*cos(A.im), e*sin(A.im));
}

template<class T> inline complex<T> cos(complex<T> const &A){
  complex<T> J(0., 1.);
  return (exp(J * A) + exp(-J * A)) / 2.;
}

template<class T> inline complex<T> sin(complex<T> const &A){
  complex<T> J(0., 1.);
  return (exp(J * A) - exp(-J * A)) / 2.;
}

template<class T> inline void set(complex<T> &A, T imm, T ree){
  A.im = imm;
  A.re = ree;
}

template<class T> inline complex<T> operator*(const complex<T> &A, const T &B){
  return complex<T> (B* (A.re), B* (A.im));
}

template<class T> inline complex<T> operator*(const T &B, const complex<T> &A){
  return complex<T> (B* (A.re), B* (A.im));
}

template<class T> inline complex<T> operator/(const complex<T> &A, const T &B){
  return complex<T> (A.re / B, A.im / B);
}

template<class T> inline complex<T> operator+(const complex<T> &A, const T &B){
  return complex<T> (B + (A.re), A.im);
}

template<class T> inline complex<T> operator+(const T &B, const complex<T> &A){
  return complex<T> (B+(A.re), A.im);
}

template<class T> inline complex<T> operator-(const complex<T> &A, const T &B){
  return complex<T> ((A.re)-B, A.im);
}

template<class T> inline complex<T> operator-(const T &B, const complex<T> &A){
  return complex<T> (B-(A.re), -A.im);
}

template<class T> struct vec2{
  inline vec2(T A, T B): a(A), b(B){
  }

  inline vec2<T>(const vec2<T> &A): a(A.a), b(A.b){
  }

  inline vec2<T>(): a(), b(){
    a.zero(); b.zero();
  }

  inline void operator = (const vec2 &A){
    a = A.a;
    b = A.b;
  }

  T a; T b;

  inline void normalize(){
    TYPE norm = sqrt(abs(a)*abs(a) + abs(b)*abs(b));
    a.re /= norm;
    a.im /= norm;
    b.re /= norm;
    b.im /= norm;
  }

  inline void unit(){
    normalize();
  }
};

template<class T> inline ostream & operator<<(ostream &output, const vec2<T> A){
  output << "["<< A.a << "]" <<endl
  << "["<<A.b << "]" <<endl;
  return output;
}

template<class T> inline vec2<T> operator+(const vec2<T> &A, const vec2<T> &B){
  vec2<T> ret(B.a+A.a, B.b+A.b);
  return ret;
}

template<class T> inline vec2<T> operator-(const vec2<T> &A, const vec2<T> &B){
  vec2<T> ret(A.a-B.a, A.b-B.b);
  return ret;
}

template<class T> inline vec2<T> operator-(const vec2<T> &B){
  vec2<T> ret(-B.a, -B.b);
  return ret;
}

template<class T> inline vec2<T> conj(vec2<T> const &A){
  vec2<T> ret(conj(A.a), conj(A.b));
  return ret;
}

template<class T> struct mat2{
  T a; T b; T c; T d;

  inline void operator = (const mat2 &A){
    a=A.a; b=A.b; c=A.c; d=A.d;
  }

  inline mat2<T>(): a(0), b(0), c(0), d(0){
  }

  inline mat2(T A, T B, T C, T D ): a(A), b(B), c(C), d(D){
  }

  inline mat2<T>(const mat2<T> &A): a(A.a), b(A.b), c(A.c), d(A.d){
  }

  inline mat2<T>(const vec2<T> &A, const vec2<T> &B): a(A.a), b(B.a), c(A.b), d(B.b){
  }
};

template<class T> inline ostream & operator<<(ostream &output, const mat2<T> A){
  output << "["<< A.a << " " <<A.b << "]" <<endl
  << "["<<A.c << " "<<A.d << "]" <<endl;
  return output;
}

template<class T> inline mat2<T> operator+(const mat2<T> &A, const mat2<T> &B){
  mat2<T> ret( B.a+A.a, B.b+A.b, B.c+A.c, B.d+A.d );
  return ret;
}

template<class T> inline mat2<T> operator-(const mat2<T> &A, const mat2<T> &B){
  mat2<T> ret(A.a-B.a, A.b-B.b, A.c-B.c, A.d-B.d);
  return ret;
}
template<class T> inline mat2<T> operator-(const mat2<T> &B){
  mat2<T> ret(-B.a, -B.b, -B.c, -B.d);
  return ret;
}

template<class T> inline mat2<T> operator*(const mat2<T> &A, const mat2<T> &B){
  mat2<T> ret(A.a*B.a+A.b*B.c, A.a*B.b+A.b*B.d,
  A.c*B.a+A.d*B.c, A.c*B.b+A.d*B.d);
  return ret;
}

template<class T> inline mat2<T> inv(mat2<T> const &A){
  mat2<T> ret(A.d/(A.a*A.d-A.b*A.c),
  -A.b/(A.a*A.d-A.b*A.c),
  -A.c/(A.a*A.d-A.b*A.c),
  A.a/(A.a*A.d-A.b*A.c));
  return ret;
}

template<class T> inline mat2<T> transpose(mat2<T> const &A){
  mat2<T> ret(A.a, A.c, A.b, A.d);
  return ret;
}

template<class T> inline mat2<T> conj(mat2<T> const &A){
  mat2<T> ret(conj(A.a), conj(A.b), conj(A.c), conj(A.d));
  return ret;
}

template<class T> inline mat2<T> adjoint(mat2<T> const &A){
  mat2<T> ret(conj(A.a), conj(A.c), conj(A.b), conj(A.d));
  return ret;
}

template<class T> inline vec2<T> operator*(const mat2<T> &A, const vec2<T> &B){
  vec2<T> ret(A.a * B.a + A.b * B.b, A.c * B.a + A.d * B.b);
  return ret;
}
