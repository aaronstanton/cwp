#include "su.h"
#include "fftw3.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

void pocs5d(complex *freqslice,complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float perci,float percf,float alphai,float alphaf,float bandi,float bandf,float *k_n_1,float *k_n_2,float *k_n_3,float *k_n_4);

void pocs5d(complex *freqslice,complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float perci,float percf,float alphai,float alphaf,float bandi,float bandf,float *k_n_1,float *k_n_2,float *k_n_3,float *k_n_4)
{  

  complex czero;
  int ix;
  float *mabs;  
  float *mabsiter;
  float sigma;
  float alpha;
  float band;
  int rank;
  int *n;
  int count;
  int iter;
  fftwf_plan p2;
  fftwf_plan p3;

  czero.r=czero.i=0;
  mabs = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  mabsiter = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);
  /******************************************************************************************** FX1X2 to FK1K2
  make the plan that will be used for each frequency slice
  written as a 4D transform with length =1 for two of the dimensions. 
  This is to make it easier to upgrade to reconstruction of 4 spatial dimensions. */
  rank = 4;
  n = ealloc1int(4);
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  p2 = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2, (fftwf_complex*)freqslice2, FFTW_FORWARD, FFTW_ESTIMATE);
  /********************************************************************************************/
  
  /******************************************************************************************** FK1K2 to FX1X2
  make the plan that will be used for each frequency slice
  written as a 4D transform with length =1 for two of the dimensions. 
  This is to make it easier to upgrade to reconstruction of 4 spatial dimensions. */
  p3 = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2, (fftwf_complex*)freqslice2, FFTW_BACKWARD, FFTW_ESTIMATE);
  /********************************************************************************************/

  fftwf_execute(p2); /* FFT x to k */
  
  /* threshold in k */
  for (ix=0;ix<nk;ix++) mabs[ix]=rcabs(freqslice2[ix]);
  fftwf_execute(p3); /* FFT k to x */
  
  for (ix=0;ix<nk;ix++) freqslice2[ix]=crmul(freqslice2[ix],1/(float) nk);
  for (iter=1;iter<Iter;iter++){  /* loop for thresholding */
    fftwf_execute(p2); /* FFT x to k */
    
    count = 0;
    
    /* This is to increase the thresholding within each internal iteration */
    sigma=quest(perci - (iter-1)*((perci-percf)/(Iter-1)),nk,mabs);
    
    /* This is to increase alpha at each iteration */
    alpha=alphai + (iter-1)*((alphaf-alphai)/(Iter-1));
    
    /* This is to increase band at each iteration */
    band=bandi + (iter-1)*((bandf-bandi)/(Iter-1));
    
    for (ix=0;ix<nk;ix++) mabsiter[ix]=rcabs(freqslice2[ix]);
    for (ix=0;ix<nk;ix++){
      /* thresholding */
      if (mabsiter[ix]<sigma) freqslice2[ix] = czero;
      /* band limitation */
      if ((k_n_1[ix] > band) || (k_n_2[ix] > band) || (k_n_3[ix] > band) || (k_n_4[ix] > band)) freqslice2[ix] = czero;
      else{ count++; }
    }
    
    fftwf_execute(p3); /* FFT k to x */
    
    for (ix=0;ix<nk;ix++) freqslice2[ix]=crmul(freqslice2[ix],1/((float) nx1fft*nx2fft*nx3fft*nx4fft));
    
	/* reinsertion into original data */
    for (ix=0;ix<nk;ix++) freqslice2[ix]=cadd(crmul(freqslice[ix],alpha),crmul(freqslice2[ix],1-alpha*wd[ix])); /* x,w */
    
  }
  
  fftwf_destroy_plan(p2);
  fftwf_destroy_plan(p3);
  
  return;

}


