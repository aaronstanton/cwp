#include "su.h"
#include "cwp.h"
#include "fftw3.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

float cgdot(complex *x,int nm);
float max_abs(complex *x, int nm);
void cg_irls(complex *m,int nm,
	     complex *d,int nd,
	     complex *m0,int nm0,
	     float *wm,int nwm,
	     float *wd,int nwd,
	     int *N,int rank,
	     int itmax_external,
	     int itmax_internal,
	     int verbose);
		   
void fft_op(complex *m,int nm,
	   complex *d,int nd,
           float *wm,int nwm,
           float *wd,int nwd,
           fftwf_plan my_plan,
           int fwd,
	   int verbose);

float max_abs(complex *x, int nm)
{   
  /* Compute Mx = max absolute value of complex vector x */	
  int i;
  float Mx;
  
  Mx = 0;
  for (i=0;i<nm;i++){  
    if(Mx<rcabs(x[i])) Mx=rcabs(x[i]);
  }
  return(Mx);
}

float cgdot(complex *x,int nm)
{
  /*     Compute the inner product */
  /*     dot=(x,x) for complex x */     
  int i;
  float  cgdot; 
  complex val;
  
  val.r=0;
  val.i=0;
  for (i=0;i<nm;i++){  
    val = cadd(val,cmul(conjg(x[i]),x[i]));
  }
  cgdot= val.r;
  return(cgdot);
}

void cg_irls(complex *m,int nm,
	     complex *d,int nd,
	     complex *m0,int nm0,
	     float *wm,int nwm,
	     float *wd,int nwd,
	     int *N,int rank,
	     int itmax_external,
	     int itmax_internal,
	     int verbose)
/* 
   Non-quadratic regularization with CG-LS. The inner CG routine is taken from
   Algorithm 2 from Scales, 1987. Make sure linear operator passes the dot product.
   In this case (MWNI), the linear operator is the FFT implemented using the FFTW package.
*/

{
  complex czero,*v,*Pv,*Ps,*s,*ss,*g,*r;
  float alpha,beta,delta,gamma,gamma_old,*P,Max_m; 
  int i,j,k,fwd,adj;
  fftwf_plan p_fwd;
  fftwf_plan p_adj;

  fwd=1;adj=0;
  czero.r=czero.i=0;
  v  = ealloc1complex(nm);
  P  = ealloc1float(nm);
  Pv = ealloc1complex(nm);
  Ps = ealloc1complex(nm);
  g  = ealloc1complex(nm);
  r  = ealloc1complex(nd);
  s = ealloc1complex(nm);
  ss = ealloc1complex(nd);
  p_fwd = fftwf_plan_dft(rank,N,(fftwf_complex*)v,(fftwf_complex*)v,FFTW_BACKWARD,FFTW_ESTIMATE);
  p_adj = fftwf_plan_dft(rank,N,(fftwf_complex*)v,(fftwf_complex*)v,FFTW_FORWARD,FFTW_ESTIMATE);

  for (i=0;i<nm;i++){
    m[i] = m0[i];				
    P[i] = 1;
    v[i] = m[i];
  }

  for (i=0;i<nd;i++){
    r[i] = d[i];				
  }

  for (j=1;j<=itmax_external;j++){
    for (i=0;i<nm;i++) Pv[i] = crmul(v[i],P[i]);
    fft_op(Pv,nm,r,nd,wm,nm,wd,nd,p_fwd,fwd,verbose);
    for (i=0;i<nd;i++) r[i] = csub(d[i],r[i]);
    fft_op(g,nm,r,nd,wm,nm,wd,nd,p_adj,adj,verbose);
    for (i=0;i<nm;i++){
      g[i] = crmul(g[i],P[i]);
      s[i] = g[i];
    }
    gamma = cgdot(g,nm);
    gamma_old = gamma;
    for (k=1;k<=itmax_internal;k++){
      for (i=0;i<nm;i++) Ps[i] = crmul(s[i],P[i]);
      fft_op(Ps,nm,ss,nd,wm,nm,wd,nd,p_fwd,fwd,verbose);
      delta = cgdot(ss,nd);
      alpha = gamma/(delta + 0.00000001);
      for (i=0;i<nm;i++) v[i] = cadd(v[i],crmul(s[i],alpha));
      for (i=0;i<nd;i++) r[i] = csub(r[i],crmul(ss[i],alpha));
      fft_op(g,nm,r,nd,wm,nm,wd,nd,p_adj,adj,verbose);
      for (i=0;i<nm;i++) g[i] = crmul(g[i],P[i]);
      gamma = cgdot(g,nm);
      beta = gamma/(gamma_old + 0.00000001);
      gamma_old = gamma;
      for (i=0;i<nm;i++) s[i] = cadd(g[i],crmul(s[i],beta));      
    }
    for (i=0;i<nm;i++){ 
      m[i] = crmul(v[i],P[i]);
    }
    Max_m = max_abs(m,nm);
    
    for (i=0;i<nm;i++){  
      P[i] = rcabs(crmul(m[i],1/Max_m));
    }
  }

  free1complex(v);
  free1float(P);
  free1complex(Pv);
  free1complex(Ps);
  free1complex(s);
  free1complex(ss);
  free1complex(g);
  free1complex(r);
  fftwf_destroy_plan(p_fwd);
  fftwf_destroy_plan(p_adj);
  return;
  
}

void fft_op(complex *m,int nm,
	   complex *d,int nd,
           float *wm,int nwm,
           float *wd,int nwd,
           fftwf_plan my_plan,
           int fwd,
	   int verbose)
/* forward and adjoint fft operator for mwni*/
{
  int i;
  if (fwd){    
    for (i=0;i<nm;i++)  m[i] = crmul(m[i],wm[i]);
    fftwf_execute_dft(my_plan, (fftwf_complex*)m, (fftwf_complex*)d);
    for (i=0;i<nd;i++)  d[i] = crmul(d[i],wd[i]/sqrt((float) nm));
  }
  else{
    for (i=0;i<nd;i++)  d[i] = crmul(d[i],wd[i]);
    fftwf_execute_dft(my_plan, (fftwf_complex*)d, (fftwf_complex*)m);
    for (i=0;i<nm;i++)  m[i] = crmul(m[i],wm[i]/sqrt((float) nm));
  }
  return;
}

