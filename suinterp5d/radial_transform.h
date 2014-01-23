#include "su.h"
#include "cwp.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif


void radial_transform(complex *m,int nm,
	              complex *d,int nd,
                      float *wm,int nwm,
                      float *wd,int nwd,
                      int fwd,
	              int verbose);

void radial_transform(complex *m,int nm,
	              complex *d,int nd,
                      float *wm,int nwm,
                      float *wd,int nwd,
                      int fwd,
	              int verbose)
/* forward and adjoint radial-transform operator*/
{
  /* see http://en.wikipedia.org/wiki/N-sphere */
  if (fwd){
    for (i=0;i<nd;i++){    
      x1[i] = r[i]*cos(phi1[i]);
      x2[i] = r[i]*sin(phi1[i])*cos(phi2[i]);
      x3[i] = r[i]*sin(phi1[i])*sin(phi2[i])*cos(phi3[i]);
      x4[i] = r[i]*sin(phi1[i])*sin(phi2[i])*sin(phi3[i])*cos(phi4[i]);
      x5[i] = r[i]*sin(phi1[i])*sin(phi2[i])*sin(phi3[i])*sin(phi4[i]);
    }
  }
  else{
    for (i=0;i<nd;i++){    
      r[i]    =   sqrt(x1[i]*x1[i] + x2[i]*x2[i] + x3[i]*x3[i] + x4[i]*x4[i] + x5[i]*x5[i]);
      phi1[i] =   arccot(x1[i]/sqrt(x5[i]*x5[i] + x4[i]*x4[i] + x3[i]*x3[i] + x2[i]*x2[i]));
      phi2[i] =   arccot(x2[i]/sqrt(x5[i]*x5[i] + x4[i]*x4[i] + x3[i]*x3[i]));
      phi3[i] =   arccot(x3[i]/sqrt(x5[i]*x5[i] + x4[i]*x4[i]));
      phi4[i] = 2*arccot((sqrt(x5[i]*x5[i] + x4[i]*x4[i]) + x4[i])/x5[i]);
    }
  }
  return;
}

