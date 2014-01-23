#include "su.h"
#include "fftw3.h"
#include "mwni5d.h"
#include "pocs5d.h"
#include "seqsvd5d.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

void process1c(float **datain,float **dataout,int verbose,int nt,int nx,float dt,int *x1h,int *x2h,int *x3h,int *x4h,int nx1,int nx2,int nx3,int nx4,float *wd_no_pad,int iter,int iter_external,float alphai,float alphaf,float bandi,float bandf,float fmax,int method,int ranki,int rankf);

void process1c(float **datain,float **dataout,int verbose,int nt,int nx,float dt,int *x1h,int *x2h,int *x3h,int *x4h,int nx1,int nx2,int nx3,int nx4,float *wd_no_pad,int iter,int iter_external,float alphai,float alphaf,float bandi,float bandf,float fmax,int method,int ranki,int rankf)
{  
  int it, ix, iw;
  complex czero;
  int ntfft,nx1fft,nx2fft,nx3fft,nx4fft,nw,nk;
  float perci;
  float percf;
  int padfactor;
  float *wd;
  float **pfft; 
  complex **cpfft;
  int N; 
  complex *out;
  fftwf_plan p1;
  int *mapping_vector;
  int ix_no_pad;
  float *k_n_1;  
  float *k_n_2;  
  float *k_n_3;  
  float *k_n_4; 
  int sum_wd;
  int ix1,ix2,ix3,ix4;
  float  f_low;
  float  f_high;
  int if_low;
  int if_high;
  float *out2;
  fftwf_plan p4;
  complex* in2;
  complex* freqslice;
  float* in;
  complex* freqslice2;

  czero.r=czero.i=0;
  perci = 0.999;
  percf = 0.001;
  padfactor = 2;
  /* copy data from input to FFT array and pad with zeros */
  ntfft = npfar(padfactor*nt);
  /* DANGER: YOU MIGHT WANT TO PAD THE SPATIAL DIRECTIONS TOO.*/
  nx1fft = nx1;
  nx2fft = nx2;
  nx3fft = nx3;
  nx4fft = nx4;
  if(nx1==1) nx1fft = 1;
  if(nx2==1) nx2fft = 1;
  if(nx3==1) nx3fft = 1;
  if(nx4==1) nx4fft = 1;
  nw=ntfft/2+1;
  nk=nx1fft*nx2fft*nx3fft*nx4fft;
  wd = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);
  freqslice = ealloc1complex(nx1fft*nx2fft*nx3fft*nx4fft);
  
  if (nx > nx1*nx2*nx3*nx4) {
  pfft  = ealloc2float(ntfft,nx); /* trace oriented (Hale's reversed convention for alloc)*/
  cpfft = ealloc2complex(nw,nx);     /* trace oriented*/
  }
  else{
  pfft  = ealloc2float(ntfft,nx1*nx2*nx3*nx4); /* trace oriented (Hale's reversed convention for alloc)*/
  cpfft = ealloc2complex(nw,nx1*nx2*nx3*nx4);     /* trace oriented*/
  }
  /* copy data from input to FFT array and pad with zeros in time dimension*/
  for (ix=0;ix<nx;ix++){
    for (it=0; it<nt; it++) pfft[ix][it]=datain[ix][it];
    for (it=nt; it< ntfft;it++) pfft[ix][it] = 0.0;
  }
  /******************************************************************************************** TX to FX
  transform data from t-x to w-x using FFTW */
  N = ntfft; 
  out = ealloc1complex(nw);
  in = ealloc1float(N);
  p1 = fftwf_plan_dft_r2c_1d(N, in, (fftwf_complex*)out, FFTW_ESTIMATE);

  for (ix=0;ix<nx;ix++){
    for(it=0;it<ntfft;it++){
      in[it] = pfft[ix][it];
    }
    fftwf_execute(p1); /* take the FFT along the time dimension */
    for(iw=0;iw<nw;iw++){
      cpfft[ix][iw] = out[iw]; 
    }
  }
  fftwf_destroy_plan(p1);
  fftwf_free(in); fftwf_free(out);
  /********************************************************************************************/
  mapping_vector = ealloc1int(nk+1);
  for (ix=0;ix<nk;ix++){
	wd[ix] = 0;  
	mapping_vector[ix] = 0;
  }
	
  ix_no_pad = 0;
	
  for (ix_no_pad=0;ix_no_pad<nx;ix_no_pad++){
	ix = x1h[ix_no_pad]*(nx2fft*nx3fft*nx4fft) + x2h[ix_no_pad]*(nx3fft*nx4fft) + x3h[ix_no_pad]*(nx4fft) + x4h[ix_no_pad];
      if (wd_no_pad[ix_no_pad] > 0){
	wd[ix] = wd_no_pad[ix_no_pad];  
	mapping_vector[ix] = ix_no_pad+1;
      }	
  }
	
  /* algorithm starts*/
  freqslice2= ealloc1complex(nx1fft*nx2fft*nx3fft*nx4fft);
  
  /* make long vectors of normalized wavenumbers (for each of the 4 dimensions). */
  /* (to be used in the band limitation) */
  k_n_1 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  k_n_2 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  k_n_3 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  k_n_4 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft); 
  sum_wd = 0;
  ix        = 0;
  for (ix1=0;ix1<nx1fft;ix1++){
    for (ix2=0;ix2<nx2fft;ix2++){
      for (ix3=0;ix3<nx3fft;ix3++){
	for (ix4=0;ix4<nx4fft;ix4++){
	  if (ix1>nx1fft/2) {
	    k_n_1[ix] = 1 - (float) (ix1-nx1fft/2)/(nx1fft/2);
	  }
	  else {
	    k_n_1[ix] = 1 - (float) (nx1fft/2-ix1)/(nx1fft/2);
	  }
	  if (ix2>nx2fft/2) {
	    k_n_2[ix] = 1 - (float) (ix2-nx2fft/2)/(nx2fft/2);
	  }
	  else {
	    k_n_2[ix] = 1 - (float) (nx2fft/2-ix2)/(nx2fft/2);
	  }
	  if (ix3>nx3fft/2) {
	    k_n_3[ix] = 1 - (float) (ix3-nx3fft/2)/(nx3fft/2);
	  }
	  else {
	    k_n_3[ix] = 1 - (float) (nx3fft/2-ix3)/(nx3fft/2);
	  }
	  if (ix4>nx4fft/2) {
	    k_n_4[ix] = 1 - (float) (ix4-nx4fft/2)/(nx4fft/2);
	  }
	  else {
	    k_n_4[ix] = 1 - (float) (nx4fft/2-ix4)/(nx4fft/2);
	  }
	  if (wd[ix]>0) sum_wd++;
	  ix++;
	}
      }
    }
  }
  if (verbose) fprintf(stderr,"out of %d input traces %d traces fall in repeated bins (%f %%).\n",nx,nx-sum_wd,(float) 100*(nx-sum_wd)/nx);
  if (verbose) fprintf(stderr,"the block has %f %% missing traces.\n", (float) 100 - 100*sum_wd/(nx1fft*nx2fft*nx3fft*nx4fft));

  f_low = 0.1;   /* min frequency to process */
  f_high = fmax; /* max frequency to process */

  if(f_low>0){ 
    if_low = trunc(f_low*dt*ntfft);
  }
  else{
    if_low = 0;
  }
  if(f_high*dt*ntfft<nw){ 
    if_high = trunc(f_high*dt*ntfft);
  }
  else{
    if_high = 0;
  }

  /* freq by freq algorithm */
  for (iw=if_low;iw<if_high;iw++){
    fprintf(stderr,"\r                                         ");
    fprintf(stderr,"\rfrequency slice %d of %d",iw-if_low+1,if_high-if_low);
	  
    for (ix=0;ix<nk;ix++){
      freqslice[ix] = freqslice2[ix] = czero;	
    }

    for (ix_no_pad=0;ix_no_pad<nx;ix_no_pad++){
      ix = x1h[ix_no_pad]*(nx2fft*nx3fft*nx4fft) + x2h[ix_no_pad]*(nx3fft*nx4fft) + x3h[ix_no_pad]*(nx4fft) + x4h[ix_no_pad];
      if (wd_no_pad[ix_no_pad] > 0){
	freqslice[ix] = freqslice2[ix] = cpfft[ix_no_pad][iw];
      }	
    }

    /* The reconstruction engine: */
    if (method==1){
      pocs5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter,perci,percf,alphai,alphaf,bandi,bandf,k_n_1,k_n_2,k_n_3,k_n_4);
    }
    if (method==2){
      mwni5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter_external,iter,verbose);
    }
    if (method==3){
      seqsvd5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter,alphai,alphaf,ranki,rankf);
    }

    ix        = 0;
    ix_no_pad = 0;
    for (ix1=0;ix1<nx1fft;ix1++){
      for (ix2=0;ix2<nx2fft;ix2++){
    	for (ix3=0;ix3<nx3fft;ix3++){
    	  for (ix4=0;ix4<nx4fft;ix4++){
    	    if (ix1<nx1 && ix2<nx2 && ix3<nx3 && ix4<nx4){
    	      cpfft[ix_no_pad][iw] = freqslice2[ix];
    	      ix_no_pad++;
    	    }
	    ix++;
    	  }
    	}
      }
    }


  }

  /* zero all other frequencies */
  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++){
    for (iw=if_high;iw<nw;iw++){
      cpfft[ix][iw] = czero;
    }
  }

  free1complex(freqslice2);
  free1complex(freqslice);
  free1float(wd);
  /* algorithm ends */


  /******************************************************************************************** FX to TX
  transform data from w-x to t-x using IFFTW*/
  N = ntfft; 
  out2 = ealloc1float(ntfft);
  in2 = ealloc1complex(N);
  p4 = fftwf_plan_dft_c2r_1d(N, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++){
    /*    fprintf(stderr,"ix=%d\n", ix); */
    for(iw=0;iw<nw;iw++){

      in2[iw] = cpfft[ix][iw];
    }
    fftwf_execute(p4); /* take the FFT along the time dimension */
    for(it=0;it<nt;it++){
      pfft[ix][it] = out2[it]; 
    }
  }
  fprintf(stderr,"\n");

  fftwf_destroy_plan(p4);
  fftwf_free(in2); fftwf_free(out2);
  /********************************************************************************************/
  /* Fourier transform w to t */

  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++) for (it=0; it<nt; it++) dataout[ix][it]=pfft[ix][it]/ntfft;
  
  free2float(pfft);
  free2complex(cpfft);
  return;

}

