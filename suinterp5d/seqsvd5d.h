#include "cwp.h"
#include "su.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

/**************************************************************************************************************************************/
void cgesdd_(const char *jobz,const int *M,const int *N,complex *Avec,const int *lda,float *Svec,complex *Uvec,const int *ldu,complex *VTvec,const int *ldvt,complex *work,const int *lwork,float *rwork,int *iwork,int *info);	
void unfold(complex *in, complex **out,int *n,int a);
void fold(complex **in, complex *out,int *n,int a);
void csvd(complex **A,complex **U, float *S,complex **VT,int M,int N);
void mult_svd(complex **A, complex **U, float *S, complex **VT, int M, int N, int rank);
void seqsvd5d(complex *freqslice,complex *freqslice2,float *wd,int nx1fft, int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float alphai,float alphaf,int ranki, int rankf);
void seqsvd5d(complex *freqslice,complex *freqslice2,float *wd,int nx1fft, int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float alphai,float alphaf,int ranki, int rankf)
{  

  complex czero;
  int ix, rank; 
  int M, N;
  int *n;
  complex **uf1;  
  complex **uf2;  
  complex **uf3;
  complex **uf4;
  complex **U1;
  complex **U2;
  complex **U3;
  complex **U4;
  complex **VT1;
  complex **VT2;
  complex **VT3;
  complex **VT4;
  float *S1;
  float *S2;
  float *S3;
  float *S4;
  float alpha;
  int iter;
 
  czero.r=czero.i=0;
  n = ealloc1int(4);
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  uf1 = ealloc2complex(nx2fft*nx3fft*nx4fft,nx1fft);  
  uf2 = ealloc2complex(nx1fft*nx3fft*nx4fft,nx2fft);  
  uf3 = ealloc2complex(nx1fft*nx2fft*nx4fft,nx3fft);  
  uf4 = ealloc2complex(nx1fft*nx2fft*nx3fft,nx4fft);
  M = nx1fft;
  N = nx2fft*nx3fft*nx4fft;
  U1 = ealloc2complex(M,M);
  S1 = ealloc1float(M);
  VT1 = ealloc2complex(N,M);
  M = nx2fft;
  N = nx1fft*nx3fft*nx4fft;
  U2 = ealloc2complex(M,M);
  S2 = ealloc1float(M);
  VT2 = ealloc2complex(N,M);
  M = nx3fft;
  N = nx1fft*nx2fft*nx4fft;
  U3 = ealloc2complex(M,M);
  S3 = ealloc1float(M);
  VT3 = ealloc2complex(N,M);
  M = nx4fft;
  N = nx1fft*nx2fft*nx3fft;
  U4 = ealloc2complex(M,M);
  S4 = ealloc1float(M);
  VT4 = ealloc2complex(N,M);

  for (iter=1;iter<=Iter;iter++){  /* loop for iteration */
    /*  This is to increase rank at each iteration */
    rank=ranki + (int) trunc((iter-1)*((rankf-ranki)/(Iter-1)));
    
    unfold(freqslice2,uf1,n,1);
    M = nx1fft;
    N = nx2fft*nx3fft*nx4fft;
    csvd(uf1,U1,S1,VT1,M,N);
    mult_svd(uf1,U1,S1,VT1,M,N,rank);
    fold(uf1,freqslice2,n,1);
   
    unfold(freqslice2,uf2,n,2);
    M = nx2fft;
    N = nx1fft*nx3fft*nx4fft;
    csvd(uf2,U2,S2,VT2,M,N);
    mult_svd(uf2,U2,S2,VT2,M,N,rank);
    fold(uf2,freqslice2,n,2);
    
    unfold(freqslice2,uf3,n,3);
    M = nx3fft;
    N = nx1fft*nx2fft*nx4fft;
    csvd(uf3,U3,S3,VT3,M,N);
    mult_svd(uf3,U3,S3,VT3,M,N,rank);
    fold(uf3,freqslice2,n,3);
   
    unfold(freqslice2,uf4,n,4);
    M = nx4fft;
    N = nx1fft*nx2fft*nx3fft;
    csvd(uf4,U4,S4,VT4,M,N);    
    mult_svd(uf4,U4,S4,VT4,M,N,rank);
    fold(uf4,freqslice2,n,4);
   
    /*  This is to increase alpha at each iteration */
    alpha=alphai + (iter-1)*((alphaf-alphai)/(Iter-1));
    
    /*  reinsertion into original data */
    for (ix=0;ix<nk;ix++) freqslice2[ix]=cadd(crmul(freqslice[ix],alpha),crmul(freqslice2[ix],1-alpha*wd[ix]));
  }

  free2complex(uf1);
  free2complex(uf2);
  free2complex(uf3);
  free2complex(uf4);

  free2complex(U1);
  free1float(S1);
  free2complex(VT1);

  free2complex(U2);
  free1float(S2);
  free2complex(VT2);

  free2complex(U3);
  free1float(S3);
  free2complex(VT3);

  free2complex(U4);
  free1float(S4);
  free2complex(VT4);

  return;
}

void unfold(complex *in, complex **out,int *n,int a)
{   
  /* 
  unfold a long-vector representing a 4D tensor into a matrix.	
  in  = long vector representing a tensor with dimensions (n[1],n[2],n[3],n[4])
  out = matrix that is the unfolding of "in" with dimensions (if a=1): (n[1],n[2]*n[3]*n[4])
  */
  int nx1,nx2,nx3,nx4;
  int ix1,ix2,ix3,ix4;

  nx1=n[0];
  nx2=n[1];
  nx3=n[2];
  nx4=n[3];
  
  if (a==1){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1][ix2*nx3*nx4+ix3*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    


 
  if (a==2){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix2][ix1*nx3*nx4+ix3*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    


  if (a==3){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
            out[ix3][ix1*nx2*nx4+ix2*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==4){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix4][ix1*nx2*nx3+ix2*nx3+ix3]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  return;
}

void fold(complex **in, complex *out,int *n,int a)
{   
  /* 
  fold a matrix back to a long-vector representing a 4D tensor.	
  in  = the unfolding of "out" with dimensions (if a=1): (n[1],n[2]*n[3]*n[4])
  out = long vector representing a tensor with dimensions (n[1],n[2],n[3],n[4]) 
  */
  int nx1,nx2,nx3,nx4;
  int ix1,ix2,ix3,ix4;
  
  nx1=n[0];
  nx2=n[1];
  nx3=n[2];
  nx4=n[3];

  if (a==1){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix1][ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==2){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix2][ix1*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==3){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix3][ix1*nx2*nx4+ix2*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==4){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix4][ix1*nx2*nx3+ix2*nx3+ix3];
	  }
	}
      }
    }
  }    

  return;
}

void csvd(complex **A,complex **U,float *S,complex **VT,int M,int N)
{

  complex czero;
  complex* Avec;
  int i,j,n;
  float* Svec;  
  char jobz;
  int lda;
  int ldu;
  int ldvt;
  int lwork; 
  int info;
  complex *work;
  int lrwork;
  float *rwork;  
  float *iwork;  
  complex* Uvec; 
  complex* VTvec; 
  
  
  czero.r=czero.i=0;
  Avec = ealloc1complex(M*N); 
  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      Avec[n] = A[j][i];
      n++;
    }
  }
  Uvec = ealloc1complex(M*M); 
  n = 0;
  for (i=0;i<M;i++){
    for (j=0;j<M;j++){
      Uvec[n] = czero;
      n++;
    }
  }
  VTvec = ealloc1complex(M*N); 
  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      VTvec[n] = czero;
      n++;
    }
  }
  Svec = ealloc1float(M);  
  
  jobz = 'S';
  lda   = M;
  ldu   = M;
  ldvt  = M;
  lwork = 5*M + N; 
  work = ealloc1complex(lwork); 

  /* make float array rwork with dimension lrwork (where lrwork >= min(M,N)*max(5*min(M,N)+7,2*max(M,N)+2*min(M,N)+1)) */
  lrwork = 5*M*M+ 7*M;
  rwork = ealloc1float(lrwork);  

  /* make float array iwork with dimension 8*M (where A is MxN) */
  iwork = ealloc1float(8*M);  
  cgesdd_(&jobz, &M, &N, (complex*)Avec, &lda, Svec, (complex*)Uvec, &ldu, (complex*)VTvec, &ldvt, (complex*)work, &lwork, (float*)rwork, (int*)iwork, &info);
  
  if (info != 0)  err("Error in cgesdd: info = %d\n",info);

  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      VT[j][i] = VTvec[n];
      n++;
    }
  }
  n = 0;
  for (i=0;i<M;i++){
    for (j=0;j<M;j++){
      U[j][i] = Uvec[n];
      n++;
    }
  }

  for (i=0;i<M;i++){
    S[i] = Svec[i];
  }
  
free1complex(Avec);
free1complex(VTvec);
free1complex(Uvec);
free1float(Svec);
free1complex(work);
free1float(rwork);
free1float(iwork);

  return;
}

void mult_svd(complex **A, complex **U,float *S,complex **VT,int M,int N,int rank)
{   
  int i,j,k;
  complex czero;
  complex** SVT;
  complex sum;

  SVT = ealloc2complex(N,M);
  czero.r=czero.i=0;
  sum = czero;
  
  for (i=0;i<rank;i++){
    for (j=0;j<N;j++){
      SVT[i][j] = czero;
      if (i < rank) SVT[i][j] = crmul(VT[i][j],S[i]);
    }
  }

  for (i=0;i<M;i++){
    for (j=0;j<N;j++){
      sum = czero;
      for (k=0;k<rank;k++){
      sum = cadd(sum,cmul(SVT[k][j],U[i][k]));
      }
      A[i][j] = sum;
    }
  }

 free2complex(SVT);
  return;
}

