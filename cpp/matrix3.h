/* matrix3.h: representations of 3x3 matrices, 3x3 hermitian matrices, and 3d vectors

by Ash Richardson, Senior Data Scientist, BC Wildfire Service */
#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix2.h"
#include"misc.h"

#define cf complex<TYPE>

#define J cf(0., 1.)
#define _zero ( cf(0.,0.) )

using namespace std;

template<class T> struct vec3{
  T a; T b; T c; /* representation of 3-d vector */

  void operator = (const vec3<T> &A){
    a = A.a; b = A.b; c = A.c;
  }

  vec3<T>(T A, T B, T C) : a(A), b(B), c(C){
  }

  vec3<T>(const complex<T> &A): a(A.a), b(A.b), c(A.c){
  }

  vec3<T>(): a(_zero), b(_zero),c(_zero){
  }
};

template<class T> ostream & operator<<(ostream &output, const vec3<T> A){
  output << "transpose([" << A.a << "," << A.b << "," << A.c << "])";
  return output;
}

template<class T> TYPE norm(vec3<T> & A){
  /* l2 norm */
  return sqrt(abs(A.a)*abs(A.a) +
              abs(A.b)*abs(A.b) +
              abs(A.c)*abs(A.c));
}

template<class T> void normalize(vec3<T> & A){
  TYPE norm = sqrt(abs(A.a)*abs(A.a)+abs(A.b)*abs(A.b)+abs(A.c)*abs(A.c)); // give vector l2 length of 1
  (A.a) = (A.a) / norm;
  (A.b) = (A.b) / norm;
  (A.c) = (A.c) / norm;
}

template<class T> cf at(vec3<T> & A, int i){
  /* access vector member */
  if(i == 0) return A.a;
  else if(i == 1) return A.b;
  else if(i == 2) return A.c;
  else err("index must be in range 0-2");
  return _zero;
}

template<class T> void vset(vec3<T> & A, int i, cf dat){
  /* accessor */
  if(i == 0) A.a = dat;
  else if(i == 1) A.b = dat;
  else if(i == 2) A.c = dat;
  else err("index must be in range 0-2");
}

template<class T> vec3<T> operator+(const vec3<T> & A, const vec3<T> & B){
  return vec3<T>(B.a + A.a, B.b + A.b, B.c + A.c);
}

template<class T> vec3<T> operator-(const vec3<T> & A, const vec3<T> & B){
  return vec3<T>(A.a - B.a, A.b - B.b, A.c - B.c);
}

template<class T> vec3<T> operator-(const vec3<T> & B){
  return vec3<T>(-B.a, -B.b, -B.c);
}

template<class T> vec3<T> conj(const vec3<T> & A){
  return vec3<T>(conj(A.a), conj(A.b), conj(A.c));
}

template<class T> vec3<T> operator*(const T &B, const vec3<T> &A){
  return vec3<T> (B * (A.a), B * (A.b), B * (A.c));
}

template<class T> vec3<T> operator*(const vec3<T> &A, const T &B){
  return vec3<T> (B * (A.a), B * (A.b), B * (A.c));
}

template<class T> vec3<T> operator / (const vec3<T> &A, const T &B){
  return vec3<T> ((A.a) / B, (A.b) / B, (A.c) / B);
}

template<class T> struct matrix3{

  matrix3<T>(T A, T B, T C, T D, T E, T F, T G, T H, T I) : a(A), b(B), c(C), d(D), e(E), f(F), g(G), h(H), i(I){
  }

  matrix3<T>(const complex<T> &A): a(A.a), b(A.b), c(A.c), d(A.d), e(A.e), f(A.f), g(A.g), h(A.h), i(A.i){
  }

  matrix3<T>(){
    zero();
  }

  void zero(){
    a.real(0.); a.imag(0.); b.real(0.); b.imag(0.);
    c.real(0.); c.imag(0.); d.real(0.); d.imag(0.);
    e.real(0.); e.imag(0.); f.real(0.); f.imag(0.);
    g.real(0.); g.imag(0.); h.real(0.); h.imag(0.);
    i.real(0.); i.imag(0.);
  }

  void operator = (const matrix3 &A){
    a = A.a; b = A.b; c = A.c;
    d = A.d; e = A.e; f = A.f;
    g = A.g; h = A.h; i = A.i;
  }

  cf det(){
    return (a*e*i) - (a*f*h) - (d*b*i) + (d*c*h) + (g*b*f) - (g*c*e);
  }

  cf trace(){
    return a + e + i;
  }

  matrix3<T> inv(){
    cf dt = 1. / det();
    return matrix3<T>(dt * (e*i-f*h),  dt * (-b*i+c*h), dt * (b*f-c*e),
                      dt * (-d*i+f*g), dt * (a*i-g*c),  dt * (-a*f+c*d),
                      dt * (h*d-e*g),  dt * (-a*h+b*g), dt * (-b*d+a*e));
  }

  TYPE norm(){
    /* frobenius type norm */
    return abs(a) + abs(b) + abs(c) + abs(d) + abs(e) + abs(f) +abs(g) + abs(h) + abs(i);
  }

  T a; T b; T c; T d; T e; T f; T g; T h; T i;
};

template<class T> struct herm3{

  herm3<T>( T A, T B, T C, T D, T E, T F) : a(A), b(B), c(C), d(D), e(E), f(F){
    /*
    A = T11
    B = T12
    C = T13
    D = T22
    E = T23
    F = T33
    */
  }

  herm3<T>(const complex<T> &A): a(A.a), b(A.b), c(A.c), d(A.d), e(A.e), f(A.f){
  }

  herm3<T>(){
    zero();
  }

  void initT3(float T11R, float T12I, float T12R, float T13I, float T13R,
              float T22R, float T23I, float T23R, float T33R){
    a.real((double)T11R); a.imag( 0.);
    b.real((double)T12R); b.imag((double)T12I);
    c.real((double)T13R); c.imag((double)T13I);
    d.real((double)T22R); d.imag(0.);
    e.real((double)T23R); e.imag((double)T23I);
    f.real((double)T33R); f.imag(0.);
  }

  void zero(){
    a.real(0.); a.imag(0.);
    b.real(0.); b.imag(0.);
    c.real(0.); c.imag(0.);
    d.real(0.); d.imag(0.);
    e.real(0.); e.imag(0.);
    f.real(0.); f.imag(0.);
  }

  void operator = (const herm3 &A){
    a = A.a; b = A.b; c = A.c;
    d = A.d; e = A.e; f = A.f;
  }

  herm3<T> operator * (double s){
    return herm3<T>(s*a, s*b, s*c, s*d, s*e, s*f);
  }

  cf det(){
    cf retval = (cf)((a*d*f) - (a*e*conj(e)) - (conj(b)*b*f) +
                     (conj(b*e)*c) + (conj(c)*b*e) - (conj(c)*c*d));

    if(imag(retval) > 0.0000001) err("hermitian matrix must have real eigenvalues\n");
    return retval;
  }

  TYPE trace(){
    if(imag(a + d + f) > 0.0000001) err("hermitian matrix must have real eigenvalues");
    return real(a + d + f);
  }

  matrix3<T> inv(){
    cf dt = ((cf(1., 0.) / (det())));
    return matrix3<T>(dt * (d*f-abs(e)*abs(e)),
                      dt * (-b*f+c*conj(e)),
                      dt * (b*e-c*d),
                      dt * (-conj(b)*f+e*conj(c)),
                      dt * (a*f-abs(c)*abs(c)),
                      dt * (-a*e+c*conj(b)),
                      dt * (conj(b)*conj(e)-d*conj(c)),
                      dt * (-a*conj(e)+b*conj(c)),
                      dt * (a*d-abs(b)*abs(b)));
  }

  TYPE norm(){
    /* frobenius norm */
    return abs(a) + 2. * abs(b) + 2. * abs(c) + abs(d) + 2. * abs(e) + abs(f);
  }

  TYPE d3( const herm3<T> & Z){
    /* sum of positive definite matrices is positive definite: */

    if(real( det()) < 0){
      cout << "Matrix: " << *this <<endl;
      cout << "Determinant: " << det() <<endl;
      err("determinant of hermitian matrix was negative\n");
    }

    /* Eqn. 8.13, p.268 of "POLARIMETRIC RADAR IMAGING" by Dr. ERIC POTTIER and Dr. JONG-SEN LEE
    Assume this object is the class centre, Cm
    d3(Z, Cm) = ln(det(Cm)) + trace( inv(Cm)*Z ) */

    cf val = (-Z.a*d*f + Z.a*e*conj(e) + conj(Z.b)*b*f - conj(Z.b)*c*conj(e)
              -conj(Z.c)*b*e + conj(Z.c)*c*d + Z.b*conj(b)*f - Z.b*e*conj(c) - Z.d*a*f
              +Z.d*c*conj(c) + conj(Z.e)*a*e - conj(Z.e)*c*conj(b) - Z.c*conj(b)*conj(e)
              +Z.c*d*conj(c) + Z.e*a*conj(e) - Z.e*b*conj(c) - Z.f*a*d + Z.f*b*conj(b));
    val = val / (-a*d*f+a*e*conj(e)+conj(b)*b*f-conj(b)*c*conj(e)-conj(c)*b*e+conj(c)*c*d);

    if(imag(val) > 0.000001){
      printf("Warning: in formula d3() the trace should have no imaginary part.\n");
    }
    return real((det()).log()) + real(val);
  }

  TYPE Dij(herm3<T> & C){
    /* equation 8.26 of p271 of "POLARIMETRIC RADAR IMAGING" by Dr. ERIC POTTIER and JONG-SEN LEE */

    TYPE a1 = log(real((det())));
    TYPE a2 = log(real((C.det())));

    if(imag(det()) > 0.00000001) err("unreal determinant\n");

    if(imag(C.det()) > 0.00000001) err("unreal determinant\n");

    cf a3 = ((inv() * C) + ((C.inv())*(*this)) ).trace();

    if(imag(a3) > 0.000000001) err("unreal trace");

    return 0.5 * (a1 + a2 + real(a3));
  }

  T a; T b; T c; T d; T e; T f;
};

template<class T> herm3<T> operator * (double s, const herm3<T> & A){
  return herm3<T>( s*(A.a), s*(A.b), s*(A.c), s*(A.d), s*(A.e), s*(A.f));
}

template<class T> ostream & operator<<( ostream &output, const herm3<T> A){
  output      << (real(A.a) > 0 ? " ":"") << A.a  << " " << (real(A.b) > 0 ? " ":"") <<      A.b  << " " << (real(A.c) > 0 ? " ":"") << A.c << endl;
  output << (real(A.b) > 0 ? " ":"") << conj(A.b) << " " << (real(A.d) > 0 ? " ":"") <<      A.d  << " " << (real(A.e) > 0 ? " ":"") << A.e << endl;
  output << (real(A.c) > 0 ? " ":"") << conj(A.c) << " " << (real(A.e) > 0 ? " ":"") << conj(A.e) << " " << (real(A.f) > 0 ? " ":"") << A.f << endl;
  return output;
}

template<class T> herm3<T> operator + ( const herm3<T> & A, const herm3<T> & B){
  return herm3<T>(B.a+A.a, B.b+A.b, B.c+A.c, B.d+A.d, B.e+A.e, B.f+A.f);
}

template<class T> matrix3<T> operator + ( const matrix3<T> & A, const matrix3<T> & B){
  return matrix3<T>(B.a+A.a, B.b+A.b, B.c+A.c,
  B.d+A.d, B.e+A.e, B.f+A.f,
  B.g+A.g, B.h+A.h, B.i+A.i);
}

template<class T> matrix3<T> operator * (const herm3<T> & A, const herm3<T> & B){
  return matrix3<T>((A.a*B.a + A.b*conj(B.b)+A.c*conj(B.c)),
                    (A.a*B.b + A.b*B.d + A.c*conj(B.e)),
                    (A.a*B.c + A.b*B.e + A.c*B.f),
                    (conj(A.b)*B.a + A.d*conj(B.b) + A.e*conj(B.c)),
                    (conj(A.b)*B.b + A.d*B.d + A.e*conj(B.e)),
                    (conj(A.b)*B.c + A.d*B.e + A.e*B.f),
                    (conj(A.c)*B.a + conj(A.e*B.b) + A.f*conj(B.c)),
                    (conj(A.c)*B.b + conj(A.e)*B.d + A.f*conj(B.e)),
                    (conj(A.c)*B.c + conj(A.e)*B.e + A.f*B.f));
}

template<class T> matrix3<T> operator * ( const matrix3<T> & X, const herm3<T> & A){
  return matrix3<T>(X.a*A.a+X.b*conj(A.b)+X.c*conj(A.c),
                    X.a*A.b+X.b*A.d+X.c*conj(A.e),
                    X.a*A.c+X.b*A.e+X.c*A.f,
                    X.d*A.a+X.e*conj(A.b)+X.f*conj(A.c),
                    X.d*A.b+X.e*A.d+X.f*conj(A.e),
                    X.d*A.c+X.e*A.e+X.f*A.f,
                    X.g*A.a+X.h*conj(A.b)+X.i*conj(A.c),
                    X.g*A.b+X.h*A.d+X.i*conj(A.e),
                    X.g*A.c+X.h*A.e+X.i*A.f);
}

template<class T> herm3<T> operator-(const herm3<T> & A, const herm3<T> & B){
  return herm3<T>(A.a-B.a, A.b-B.b, A.c-B.c, A.d-B.d , A.e-B.e, A.f-B.f );
}

template<class T> herm3<T> operator-(const herm3<T> & B){
  return herm3<T>(-B.a, -B.b, -B.c, -B.d, -B.e, -B.f);
}

template<class T> vec3<T> operator*(const herm3<T> &A, const vec3<T> &B){
  return vec3<T>(A.a*B.a + A.b*B.b + A.c*B.c,
                 A.d*B.b + A.e*B.c + B.a*conj(A.b),
                 A.f*B.c + B.a*conj(A.c) + B.b*conj(A.e));
}

vec3<cf> solve_cubic(cf a, cf b, cf c, cf d){

  /* add case to avoid div by 0 */
  TYPE _2t13 = pow(2., 0.3333333333333333);
  TYPE _2t23 = pow(2., 0.6666666666666666);
  TYPE sqrt3 = sqrt(3.);

  cf t2 = 3.*a*c -b*b;
  cf t1 = b*(-2.*b*b + 9.*a*c) - 27.*a*a*d ;
  cf t0 = (t1 + pow( 4.*(t2*t2*t2) + (t1*t1) ,0.5));

  cf t3 = pow(t0 , 0.333333333333333333333333) ;

  cf aX6 = (6.*a*t3); cf bX2 = -2.*b*t3; cf X2 = t3*t3;
  return vec3<cf>((bX2 + _2t23*X2 - 2.*_2t13*t2)/aX6 ,
                  (2.*bX2 + _2t13*(2.*(1. + J * sqrt3) * t2 + J * _2t13*(J + sqrt3)*X2 ))/(2.*aX6),
                  (2.*bX2 + _2t13*(2.*(1. - J * sqrt3) * t2 - _2t13*(1.+ J * sqrt3)*X2 ))/(2.*aX6));
}

vec3<cf> solve_characteristic(const herm3<cf> & A){

  /*solve characteristic equation for the 3x3 conj symmetric matrix: [ a b c; b* d e; c* e* f ]*/
  cf a(A.a); cf b(A.b); cf c(A.c);
  cf d(A.d); cf e(A.e); cf f(A.f);

  cf _A; cf _B; cf _C; cf _D;
  cf lambda1; cf lambda2; cf lambda3;

  _A = cf(-1.,0); //-1 + 0*I;
  _B = (a + d + f);
  _C = (-(a*d) - a*f - d*f + b*conj(b) + c*conj(c) + e*conj(e));
  _D = d*(a*f - c*conj(c)) + e*(b*conj(c) - a*conj(e)) + conj(b)*(-(b*f) + c*conj(e));
  vec3<cf> x(solve_cubic(_A, _B, _C, _D)) ;

  //cout << "characteristic residual "<< residual(x, _A, _B, _C, _D) <<endl;
  return x;
}

vec3<cf> eigv( herm3<cf> &A, cf & lambda){
  /*
  >> syms a lambda b y c z d y e z
  >> solve( '(a-lambda)+b*y+c*z', 'conj(b) + y*(d-lambda) +e*z')

  ans =
  y: [1x1 sym]
  z: [1x1 sym]
  >> */

  return vec3<cf>(cf(1.,0.),
  -((A.a)*(A.e)-lambda*(A.e)-(A.c)*conj((A.b)))/((A.b)*(A.e)-(A.d)*(A.c)+lambda*(A.c)),
  (-(A.b)*conj((A.b))-lambda*(A.a)+(A.d)*(A.a)-(A.d)*lambda+(lambda*lambda))/((A.b)*(A.e)-(A.d)*(A.c)+lambda*(A.c)));

  /* x.a = cf(1.,0.);
     x.b = -(a*e-lambda*e-c*conj(b))/(b*e-d*c+lambda*c);
     x.c = (-b*conj(b)-lambda*a+d*a-d*lambda+lambda^2)/(b*e-d*c+lambda*c); */
}

TYPE eig(herm3<cf> &A , vec3<cf> &L, vec3<cf> &E1, vec3<cf> &E2, vec3<cf> &E3){
  vec3<cf> lambdas(solve_characteristic(A));
  vec3<cf> e1(eigv(A, lambdas.a));
  vec3<cf> e2(eigv(A, lambdas.b));
  vec3<cf> e3(eigv(A, lambdas.c));

  cf l1 = lambdas.a; cf l2 = lambdas.b; cf l3 = lambdas.c;
  normalize(e1); normalize(e2); normalize(e3);

  int tmpi;
  double tmpf;

  int ind[3] = {
    0,
    1,
    2
  };

  double ABS[3] = {
    abs(l1),
    abs(l2),
    abs(l3)
  };

  vec3<cf> * ptr[3] = {
    &e1,
    &e2,
    &e3
  };

  int j,i;

  for(j = 2; j > 0; j--){
    for(i = 0; i < j; i++){
      if(ABS[i] < ABS[i + 1]){
        tmpi = ind[i];
        ind[i] = ind[i + 1];
        ind[i + 1] = tmpi;
        tmpf = ABS[i];
        ABS[i] = ABS[i + 1];
        ABS[i + 1] = tmpf;
      }
    }
  }

  /* vec3<cf> d1 = (A*e1)/l1 - e1;
     vec3<cf> d2 = (A*e2)/l2 - e2;
     vec3<cf> d3 = (A*e3)/l3 - e3;*/

  for(i = 0; i < 3; i++)
      vset(L, i, at(lambdas,ind[i]));

  E1 = *(ptr[ind[0]]);
  E2 = *(ptr[ind[1]]);
  E3 = *(ptr[ind[2]]);
  vec3<cf> d1 = (A*E1)/(L.a) - E1;
  vec3<cf> d2 = (A*E2)/(L.b) - E2;
  vec3<cf> d3 = (A*E3)/(L.c) - E3;
  return norm(d1) + norm(d2) + norm(d3);
  /* sort eigenvectors */
  //return norm ( (A*e1)/l1 - e1) + norm( (A*e2)/l2 - e2) + norm( (A*e3)/l3 - e3)
}
