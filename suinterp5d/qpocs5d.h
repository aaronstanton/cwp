#include "su.h"
#include "fftw3.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

float qabs(complex c1, complex c2);
void qpocs5d(complex *freqslice_c1,complex *freqslice_c2,complex *freqslice2_c1,complex *freqslice2_c2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float perci,float percf,float alphai,float alphaf,float bandi,float bandf,float *k_n_1,float *k_n_2,float *k_n_3,float *k_n_4);

void qpocs5d(complex *freqslice_c1,complex *freqslice_c2,complex *freqslice2_c1,complex *freqslice2_c2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float perci,float percf,float alphai,float alphaf,float bandi,float bandf,float *k_n_1,float *k_n_2,float *k_n_3,float *k_n_4)
{

  complex czero;
  int ix;
  float* mabs;  
  float* mabsiter;
  float sigma;
  float alpha;
  float band;
  int *n;
  int rank;
  fftwf_plan p2a;
  fftwf_plan p2b;
  fftwf_plan p3a;
  fftwf_plan p3b;
  int iter;
  int count;

  mabs = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  mabsiter = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);
  czero.r=czero.i=0;
  n = ealloc1int(4);  
  /********************************************************************************************* FX1X2 to FK1K2
  make the plan that will be used for each frequency slice
  written as a 4D transform with length =1 for two of the dimensions. 
  This is to make it easier to upgrade to reconstruction of 4 spatial dimensions. */
  rank = 4;
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  p2a = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2_c1, (fftwf_complex*)freqslice2_c1, FFTW_FORWARD, FFTW_ESTIMATE);
  p2b = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2_c2, (fftwf_complex*)freqslice2_c2, FFTW_FORWARD, FFTW_ESTIMATE);
  /********************************************************************************************/

  /******************************************************************************************** FK1K2 to FX1X2
  make the plan that will be used for each frequency slice
  written as a 4D transform with length =1 for two of the dimensions. 
  This is to make it easier to upgrade to reconstruction of 4 spatial dimensions. */
  p3a = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2_c1, (fftwf_complex*)freqslice2_c1, FFTW_BACKWARD, FFTW_ESTIMATE);
  p3b = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2_c2, (fftwf_complex*)freqslice2_c2, FFTW_BACKWARD, FFTW_ESTIMATE);
  /********************************************************************************************/

  fftwf_execute(p2a); /* QFT x to k */
  fftwf_execute(p2b); /* QFT x to k */
  
  /* threshold in k*/
  for (ix=0;ix<nk;ix++) mabs[ix]=qabs(freqslice2_c1[ix],freqslice2_c2[ix]);

  fftwf_execute(p3a); /* QFT k to x*/
  fftwf_execute(p3b); /* QFT k to x */
  
  for (ix=0;ix<nk;ix++){ 
    freqslice2_c1[ix] = crmul(freqslice2_c1[ix],(float) 1/nk);
    freqslice2_c2[ix] = crmul(freqslice2_c2[ix],(float) 1/nk);
  }
  for (iter=1;iter<Iter;iter++){  /* loop for thresholding */
    fftwf_execute(p2a); /* QFT x to k */
    fftwf_execute(p2b); /* QFT x to k */ 

    count = 0; /* count kept */
    
    /* This is to increase the thresholding within each internal iteration */
    sigma=quest(perci - (iter-1)*((perci-percf)/(Iter-1)),nk,mabs);
    
    /* This is to increase alpha at each iteration */
    alpha=alphai + (iter-1)*((alphaf-alphai)/(Iter-1));
    
    /* This is to increase band at each iteration */
    band=bandi + (iter-1)*((bandf-bandi)/(Iter-1));
    
    for (ix=0;ix<nk;ix++) mabsiter[ix]=qabs(freqslice2_c1[ix],freqslice2_c2[ix]);
    for (ix=0;ix<nk;ix++){
      /* thresholding */
      if (mabsiter[ix]<sigma){ 
        freqslice2_c1[ix] = czero;
        freqslice2_c2[ix] = czero;
      }
      /* band limitation */
      if ((k_n_1[ix] > band) || (k_n_2[ix] > band) || (k_n_3[ix] > band) || (k_n_4[ix] > band)){ 
	freqslice2_c1[ix] = czero;
	freqslice2_c2[ix] = czero;
      }
      else{ count++; }
    }
    
    fftwf_execute(p3a); /* QFT k to x */
    fftwf_execute(p3b); /* QFT k to x */
    
    for (ix=0;ix<nk;ix++){ 
      freqslice2_c1[ix] = crmul(freqslice2_c1[ix],(float) 1/(nx1fft*nx2fft*nx3fft*nx4fft));
      freqslice2_c2[ix] = crmul(freqslice2_c2[ix],(float) 1/(nx1fft*nx2fft*nx3fft*nx4fft));
    }
	/* reinsertion into original data */
    for (ix=0;ix<nk;ix++){ 
      freqslice2_c1[ix]=cadd(crmul(freqslice_c1[ix],alpha),crmul(freqslice2_c1[ix],1-alpha*wd[ix])); /* x,w */
      freqslice2_c2[ix]=cadd(crmul(freqslice_c2[ix],alpha),crmul(freqslice2_c2[ix],1-alpha*wd[ix])); /* x,w */
    }
  }
  
  fftwf_destroy_plan(p2a);
  fftwf_destroy_plan(p2b);
  fftwf_destroy_plan(p3a);
  fftwf_destroy_plan(p3b);
  
  return;

}

float qabs(complex c1, complex c2){
  /* The quaternion amplitude (of a quaternion that is expressed as two complex numbers) */
  float Aq;
  Aq = sqrt(c1.r*c1.r + c1.i*c1.i + c2.i*c2.i + c2.i*c2.i);
  return Aq;
}
