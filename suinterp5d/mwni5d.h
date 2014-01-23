#include "su.h"
#include "fftw3.h"
#include "cg_irls.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

void mwni5d(complex *freqslice,complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int itmax_external,int itmax_internal,int verbose);

void mwni5d(complex *freqslice,complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int itmax_external,int itmax_internal,int verbose)
{  
  complex czero;
  complex *x1;
  complex *x2;
  int rank = 4;
  int *n;
  float *wm;
  fftwf_plan prv;
  int i;
  complex* m0;

  czero.r=czero.i=0;
  wm = ealloc1float(nk);
  x1 = ealloc1complex(nk);
  x2 = ealloc1complex(nk);
  for (i=0; i<nk; i++) wm[i] = 1;
  /*********************************************************************/
  rank = 4;
  n = ealloc1int(rank);
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  prv = fftwf_plan_dft(rank,n,(fftwf_complex*)x2,(fftwf_complex*)x2,FFTW_BACKWARD,FFTW_ESTIMATE);
  /*********************************************************************/
  m0 = ealloc1complex(nk);
  for (i=0; i<nk; i++){ 
	  x1[i] = freqslice[i];
	  x2[i] = freqslice2[i];
	  m0[i] = czero;
  }
  
  cg_irls(x2,nk,
	       x1,nk,
	       m0,nk,
	       wm,nk,
	       wd,nk,
	       n,rank,
	       itmax_external,
	       itmax_internal,
	       verbose);
   
  fftwf_execute(prv); /* FFT (k to x)  (x2 ---> x1) */
  for (i=0; i<nk; i++){ 
    freqslice2[i]=crmul(x2[i],1/sqrt((float) nk));
  }	
  fftwf_destroy_plan(prv);
  free1complex(x1);
  free1complex(x2);
  return;
}

