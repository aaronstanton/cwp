/* Copyright (c) Signal Analysis and Imaging Group (SAIG), University of Alberta, 2012.*/
/* All rights reserved.                       */
/* suinterp5d  :  $Date:September     2012- Last version May 2013  */

#include "su.h"
#include "segy.h"
#include "header.h"
#include <time.h>
#include "process1c.h"
#include "process2c.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

/*********************** self documentation **********************/
char *sdoc[] = {
  " 	   							      ",
  " SUINTERP5D  5D Reconstruction using:                              ",
  "             Projection Onto Convex Sets (POCS)         (method=1) ",
  "             Minimum Weighted Norm Interpolation (MWNI) (method=2) ",
  "             Sequential SVD (SEQSVD) aka poor mans Higher Order SVD (method=3) ",
  "                                                                   ",
  "           The spatial coordinates used are:                       ",
  "	          cdp, cdpt, swdep, gwdep.                                ",
  "	          which correspond to imx,imy,ih,iaz from sugeom          ",
  "                                                                   ",
  "           User provides:                                          ",
  "             in1=datain.su                                         ",
  "             out1=dataout1.su                                      ",
  "             mc=(1 or 2) allows for vector reconstruction of mc    ",
  "                         data (at this stage only using method=1). ",
  "                         2c reconstruction is done via a quaternion",
  "                         representation in the frequency domain:   ",
  "                         C1_real(x,w)  + C1_imag(x,w)i +           ",
  "                         C2_real(x,w)j + C2_imag(x,w)k             ",
  "                         The additional components are specified   ",
  "                         using in2= & out2=                        ",
  "                                                                   ",
  "             gridfile=gridfile_name                                ",
  "                      a text file with a grid definition           ",
  "                      (for a description of the gridfile see the   ",
  "                       sugeom doc)                                 ",
  "             limfile=limfile_name                                  ",
  "                     a text file with dimensions for the block     ",
  "                     (can be generated by sugeom)                  ",
  "             trfile=trfile_name  (optional)                        ",
  "                    a binary file with trace numbers to be read,   ",
  "                    which allows a very large input file to be read",
  "                    efficiently.                                   ",
  "                    (can be generated by sugeom)                   ",
  "                                                                   ",
  "           Other parameters (values in brackets are defaults):     ",
  "             method(=1) POCS, 2 for MWNI, 3 for SEQSVD,            ",
  "             iter; number of iterations (default=100)              ",
  "                                                                   ",
  "             when method=2 is used MWNI is used, and the parameter ",
  "             iter_external(=3) can be set for precondictioned CG   ",
  "             for sparsity promotion.                               ",
  "                                                                   ",
  "             when method=3 is used Sequential SVD is used, which   ",
  "             is a similar method to higher order SVD. The 4D data  ",
  "             is unfolded to a matrix 4 different ways and SVD is   ",
  "             applied to each unfolding, then the data is reinserted",
  "             into the original data (similar to POCS). At each SVD ",
  "             the highest SVs are kept, the number of SVs kept at   ",
  "             each iteration is controlled by ranki(=5) and         ", 
  "             rankf(=5) which are the ranks at intial and final     ",
  "             iterations  (linear in between initial and final)     ",
  "                                                                   ",
  "             fmax; maximum frequency (default=Nyquist=0.5/dt)      ",
  "             alphai, alphaf; denoising parameter at initial and    ",
  "                             final iteration (default 1,1).        ",
  "                             (linear in between initial and final) ", 
  "                             1 => no noise , near 0 => noisy input ",
  "             bandi, bandf; band-limitation parameter at initial and",
  "                           final iteration (default 1,1).          ",
  "                           1 => no bandlim , near 0 => high bandlim",
  "             dec; random decimation of the input data for testing. ",
  "                  0 => no decimation (default), 1 => all zeros     ", 
  "                                                                   ",
  "             Ltw=200 length of time windows in samples             ",
  "             Dtw=10  length of overlap of time windows in samples  ",
  "             ntr=1000000 number of input traces in the block. It is",
  "             best to set this parameter as the default is set quite", 
  "             high (1e6).                                           ",
  "                                                                   ",
  "             the program updates ep with a unique value for each   ",
  "             bin (a combination of cdp & cdpt). This can be used to", 
  "             stack overlapping blocks eg:                          ",
  "             susort < d.su ep offset otrav > d_sorted.su           ",
  "             sustack < d_sorted.su key=otrav > d_merged_blocks.su  ",
  "                                                                   ",
  "   References:                                                     ",
  "                                                                   ",
  "   Method 1 (POCS):                                                ",
  "   Abma, R., and N. Kabir, 2006, 3D interpolation of irregular data", 
  "        with a POCS algorithm: Geophysics,71,no. 6,E91–E96.        ",
  "                                                                   ",
  "   Method 2 (MWNI):                                                ",
  "   Liu, B., and M. D. Sacchi, 2004, Minimum weighted norm          ",
  "        interpolation of seismic records: Geophysics,69,1560–1568. ",
  "                                                                   ",
  "   Method 3 (SEQSVD):                                              ",
  "   Kreimer, N. and M. D. Sacchi, Tensor unfolding principles and   ",
  "        applications to rank reduction reconstruction and          ",
  "        denoising: CSEG,2012                                       ",
  "                                                                   ",
  "   Method 1 for 2C data (Vector POCS):                             ",
  "   Stanton, A., and M. D. Sacchi, 2011, Multicomponent seismic data",
  "        reconstruction using the quaternion fourier transform and  ",
  "        pocs: SEG Technical Program Expanded Abstracts,            ",
  "        30,1267–1272.                                              ",
  "                                                                   ",
NULL};
/* Credits:
 * Aaron Stanton.
 * Trace header fields accessed: ns, dt, ntr, cdp, cdpt, swdep, gwdep.
 * Last changes: May : 2013 
 */
/**************** end self doc ***********************************/

segy tr;
FILE *fp1,*fp2,*fp1out,*fp2out,*fp1true,*fp2true; 

int main(int argc, char **argv)
{
  float *h;   
  int *x1;	    
  int *x2;
  int *x3;	    
  int *x4;   
  int *x1h;	    
  int *x2h;
  int *x3h;	    
  int *x4h;
  float *trnum;
  float recip, ang, omx, omy, dmx, dmy, dh, daz, gamma;
  int verbose;
  cwp_String trfile=""; /* file containing trace numbers to be read */ 	
  FILE *trfilep;

  time_t start,finish;
  double elapsed_time;
  int it,ih;
  float t0=0;
  float *Wd        = 0;
  float *t;
  int nt, ntr, nh; 
  int nx1, nx2, nx3, nx4; 
  int method;
  int plot; 
  float dt;
  float dx1,dx2,dx3,dx4;
  int iter, iter_external, ranki, rankf;
  float fmax;
  float alphai, alphaf; /* denoising parameter, 1 => no noise , near 0 => noisy input traces.*/
  float bandi, bandf; /* band-limitation parameter, 1 => no bandlim , near 0 => high bandlim.*/
  float dec;
  int mc; /* flag for one component or multiple components.*/
  int config; /* flag for configuration (1=mhaz, 2=mhxhy, 3=sg, 4=shaz, 5=shxhy, 6=ghaz, 7=ghxhy)*/
  int truedata; /* flag for comparing results with true data*/
  cwp_String in1, out1, in2, out2;
  cwp_String in1true, in2true;  
  cwp_String gridfile, limfile;
  float x1_min_tmp, x1_max_tmp, x2_min_tmp, x2_max_tmp, x3_min_tmp, x3_max_tmp, x4_min_tmp, x4_max_tmp;
  float nx1_tmp, nx2_tmp, nx3_tmp, nx4_tmp;
  int x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max;
  int Ltw; 
  int Dtw;   
  float **data1in;
  float **data1in_tw;
  float **data1out;
  float **data1out_tw;
  float **data1true;
  float **data2true;
  int Ntw;
  float **data2in;
  float **data2in_tw;
  float **data2out;
  float **data2out_tw;
  int *x1hout;
  int *x2hout;
  int *x3hout;
  int *x4hout;
  float *x1out;
  float *x2out;
  float *x3out;
  float *x4out;
  int ix1,ix2,ix3,ix4;
  float rndnum = 0;
  float* wd_no_pad;
  int twstart;
  float taper;
  int Itw;
  int itw;
  float tmp_num1;
  float tmp_denom1;
  float Q1;
  float tmp_num2;
  float tmp_denom2;
  float Q2;
  float deg2rad = PI/180;
  float rad2deg = 180/PI;
  float gammainv = 1/gamma;	 		
  float ang2;	 		
  float *imximy_out;	    
  float *mx_rot;	    
  float *my_rot;	    
  float *mx_out;	    
  float *my_out;	    
  float *h_out;	    
  float *az_out;	    
  float *hx_out;	    
  float *hy_out;	    
  float *sx_out;	    
  float *sy_out;	    
  float *gx_out;	    
  float *gy_out;	    
  FILE *fp_grid;
  FILE *fp_lim;

  /*             */
    
  fprintf(stderr,"*******SUINTERP5D*********\n");
  /* Initialize */
  initargs(argc, argv);
  requestdoc(1);

  start=time(0);    
  /* Get parameters */
  if (!getparint("mc",&mc)) mc = 1; 
  if (!getparint("truedata",&truedata)) truedata = 0;
  if (truedata >= 1){
    if (!getparstring("in1true",&in1true)) err("in1true required."); 
  	if (mc >= 2){
    if (!getparstring("in2true",&in2true)) err("in2true required.");
  	}
  }
  if (!getparstring("in1",&in1)) err("in1 required."); 
  if (!getparstring("out1",&out1)) err("out1 required."); 
  if (mc >= 2){
  if (!getparstring("in2",&in2)) err("in2 required."); 
  if (!getparstring("out2",&out2)) err("out2 required."); 
  }
  if (!getparint("method", &method)){  
    method = 1;
    fprintf(stderr,"Reconstruction using POCS\n");
  }
  if (method==2){
    fprintf(stderr,"Reconstruction using MWNI\n");
  }
  if (method==3){
    fprintf(stderr,"Reconstruction using SEQSVD\n");
  }
  if (!getparint("config",&config)){
  	config = 1; 
    fprintf(stderr,"Configuration: imx imy ih iaz\n");
  }
  if (config==2){
    fprintf(stderr,"Configuration: imx imy ihx ihy\n");
    err("config=2 not currently supported");
  }
  if (config==3){
    fprintf(stderr,"Configuration: isx isy igx igy\n");
    err("config=3 not currently supported");
  }
  if (config==4){
    fprintf(stderr,"Configuration: isx isy ih iaz (...but currently the code assumes that there is only 1 source)\n");
  }
  if (config==5){
    fprintf(stderr,"Configuration: isx isy ihx ihy\n");
    err("config=5 not currently supported");
  }
  if (config==6){
    fprintf(stderr,"Configuration: igx igy ih iaz\n");
    err("config=6 not currently supported");
  }
  if (config==7){
    fprintf(stderr,"Configuration: igx igy ihx ihy\n");
    err("config=7 not currently supported");
  }
  if (!getparint("verbose", &verbose))  verbose =1;
  if (!getparint("plot",&plot)) plot = 0;
  if (!getparint("iter",&iter)) iter = 100;
  if (!getparint("iter_external",&iter_external)) iter_external = 3;
  if (!getparint("ranki",&ranki)) ranki = 5;
  if (!getparint("rankf",&rankf)) rankf = 5;
  if (!getparfloat("alphai",&alphai)) alphai = 1;
  if (!getparfloat("alphaf",&alphaf)) alphaf = 1;
  if (!getparfloat("bandi",&bandi)) bandi = 1;
  if (!getparfloat("bandf",&bandf)) bandf = 1;
  if (!getparint("ntr",&ntr)){ 
    ntr = 1000000;
    fprintf(stderr,"warning: ntr paramater not set; using ntr=%d\n",ntr);
  }
  if (!getparstring("gridfile",&gridfile)) err("gridfile required."); 
  fp_grid=fopen(gridfile, "r");
  fscanf(fp_grid, "%f", &ang);
  fgetc(fp_grid);  /*go to the next line*/
  fscanf(fp_grid, "%f %f",&omx,&omy);
  fgetc(fp_grid);  /*go to the next line*/
  fscanf(fp_grid, "%f %f",&dmx,&dmy);
  fgetc(fp_grid);  /*go to the next line*/
  fscanf(fp_grid, "%f %f",&dh,&daz);
  fgetc(fp_grid);  /*go to the next line*/
  fscanf(fp_grid, "%f %f",&recip, &gamma);
  fclose(fp_grid);
  fprintf(stderr,"grid definition:\n ang=%f\n omx=%f, omy=%f\n dmx=%f, dmy=%f\n dh=%f, daz=%f\n recip=%f, gamma=%f\n", ang,omx,omy,dmx,dmy,dh,daz,recip,gamma);  
  if (ang<0 || ang >= 180) err("Make sure 0<=ang<180");
  if (recip==1) fprintf(stderr,"warning: using recip=1 (assuming reciprocity of sources and receivers)\n");
  
  if (!getparstring("limfile",&limfile)) err("limfile required."); 
  fp_lim=fopen(limfile, "r");
  fscanf(fp_lim, "%f %f %f", &x1_min_tmp,&x1_max_tmp,&nx1_tmp);
  fgetc(fp_lim);  /*go to the next line*/
  fscanf(fp_lim, "%f %f %f", &x2_min_tmp,&x2_max_tmp,&nx2_tmp);
  fgetc(fp_lim);  /*go to the next line*/
  fscanf(fp_lim, "%f %f %f", &x3_min_tmp,&x3_max_tmp,&nx3_tmp);
  fgetc(fp_lim);  /*go to the next line*/
  fscanf(fp_lim, "%f %f %f", &x4_min_tmp,&x4_max_tmp,&nx4_tmp);
  fclose(fp_lim);
  x1_min = (int) x1_min_tmp;
  x1_max = (int) x1_max_tmp;
  nx1 = (int) nx1_tmp;
  x2_min = (int) x2_min_tmp;
  x2_max = (int) x2_max_tmp;
  nx2 = (int) nx2_tmp;
  x3_min = (int) x3_min_tmp;
  x3_max = (int) x3_max_tmp;
  nx3 = (int) nx3_tmp;
  x4_min = (int) x4_min_tmp;
  x4_max = (int) x4_max_tmp;
  nx4 = (int) nx4_tmp;

  if (!getparint("Ltw", &Ltw))  Ltw = 200; /* length of time window in samples*/
  if (!getparint("Dtw", &Dtw))  Dtw = 10; /* overlap of time windows in samples*/

  /***********************************************************************/
  /* reading 1st component:*/
  /***********************************************************************/
  fp1 = efopen(in1, "r");
  
  trnum=ealloc1float(ntr);

  if (!getparstring("trfile",&trfile)){ 
    if(!fgettra(fp1,&tr,1)) err("can't read first trace of first file");
  }
  else{
    trfilep=fopen(trfile, "rb");
    for (ih=0; ih<ntr;ih++){
      fread(&trnum[ih], sizeof(float), 1, trfilep);
    }
    fclose(trfilep);
    if(!fgettra(fp1,&tr,trnum[1])) err("can't read first trace of first file");
  }

  if (!tr.dt) err("dt header field must be set");
  if (!tr.ns) err("ns header field must be set");
  dt = ((float) tr.dt)/1000000.0;
  nt = (int) tr.ns;
  nh = ntr;
  if (!getparfloat("fmax",&fmax)) fmax = 0.5/dt;
  fmax = MIN(fmax,0.5/dt);
  if (!getparfloat("dec",&dec)) dec = 0;

  /* Allocate memory for data and model*/
  data1in   = ealloc2float(nt,nh);
  data1out  = ealloc2float(nt,nx1*nx2*nx3*nx4);
  Wd=ealloc1float(nh);
  /* number of time windows (will be updated during first 
     iteration to be consistent with total number of time samples
     and the length of each window)*/
  Ntw = 9999;	
  data1in_tw   = ealloc2float(Ltw,nh);
  data1out_tw   = ealloc2float(Ltw,nx1*nx2*nx3*nx4);
  h=ealloc1float(nh);
  x1=ealloc1int(nh);
  x2=ealloc1int(nh);
  x3=ealloc1int(nh);
  x4=ealloc1int(nh);
  x1h=ealloc1int(nh);
  x2h=ealloc1int(nh);
  x3h=ealloc1int(nh);
  x4h=ealloc1int(nh);
		
  t=ealloc1float(nt);
  memset( (void *) h, (int) '\0', nh * FSIZE);
  /* Loop over traces*/

  if (!getparstring("trfile",&trfile)){ 

    ih=0;
    if (config==1){
      x1[ih]=(int) tr.cdp;
      x2[ih]=(int) tr.cdpt;
      x3[ih]=(int) tr.swdep;
      x4[ih]=(int) tr.gwdep;
    }
    if (config==4){
      x1[ih]=(int) 0;
      x2[ih]=(int) 0;
      x3[ih]=(int) tr.swdep;
      x4[ih]=(int) tr.gwdep;      	
    }
    memcpy((void *) data1in[ih],(const void *) tr.data,nt*sizeof(float));
    ih++;
    
    do {
      if (tr.trid==2) Wd[ih]=0;
      else  Wd[ih]=1;
      if (config==1){
      	x1[ih]=(int) tr.cdp;
      	x2[ih]=(int) tr.cdpt;
      	x3[ih]=(int) tr.swdep;
      	x4[ih]=(int) tr.gwdep;
      }
      if (config==4){
      	x1[ih]=(int) 0;
      	x2[ih]=(int) 0;
      	x3[ih]=(int) tr.swdep;
      	x4[ih]=(int) tr.gwdep;      	
      }
      memcpy((void *) data1in[ih],(const void *) tr.data,nt*sizeof(float));
      ih++;
      if (ih > nh) err("Number of traces > %d\n",nh); 
    } while (fgettr(fp1,&tr));
    nh=ih;
      fprintf(stderr,"done reading 1st component: ih=%d\n",ih);

  }
  else {
    ih = 0;
    do {
      if (tr.trid==2) Wd[ih]=0;
      else  Wd[ih]=1;
      if (config==1){
      x1[ih]=(int) tr.cdp;
      x2[ih]=(int) tr.cdpt;
      x3[ih]=(int) tr.swdep;
      x4[ih]=(int) tr.gwdep;
      }
      if (config==4){
      	x1[ih]=(int) 0;
      	x2[ih]=(int) 0;
      	x3[ih]=(int) tr.swdep;
      	x4[ih]=(int) tr.gwdep;      	
      }
      memcpy((void *) data1in[ih],(const void *) tr.data,nt*sizeof(float));
      ih++;
      if (ih > nh) err("Number of traces > %d\n",nh); 
    } while (fgettra(fp1,&tr,(int) trnum[ih]) && (trnum[ih]>0));
    nh=ih;
  }
  fclose(fp1);
  /***********************************************************************/
  /* end reading the 1st component*/
  /***********************************************************************/
  if (mc >= 2){
    /***********************************************************************/
    /* reading 2nd component: (if mc >=2 )*/
    /***********************************************************************/
    fp2 = efopen(in2, "r");
    if(!fgettra(fp2,&tr,1)) err("can't read first trace of second file");

    /* Allocate memory for data and model*/
    data2in   = ealloc2float(nt,nh);
    data2out   = ealloc2float(nt,nx1*nx2*nx3*nx4);
    data2in_tw   = ealloc2float(Ltw,nh);
    data2out_tw   = ealloc2float(Ltw,nx1*nx2*nx3*nx4);
    /* Loop over traces*/

    ih=0;
    memcpy((void *) data2in[ih],(const void *) tr.data,nt*sizeof(float));
    ih++;

    if (!getparstring("trfile",&trfile)){ 
      do {
      	memcpy((void *) data2in[ih],(const void *) tr.data,nt*sizeof(float));
      	ih++;
      } while (fgettr(fp2,&tr));
      fprintf(stderr,"done reading 2nd component: ih=%d\n",ih);
      nh=ih;
    }
    else {
      ih = 0;
      do {
	memcpy((void *) data2in[ih],(const void *) tr.data,nt*sizeof(float));
	ih++;
	if (ih > nh) err("Second file: Number of traces > %d\n",nh+1); 
      } while (fgettra(fp2,&tr,(int) trnum[ih]) && (trnum[ih]>0));
      if (ih != nh)  err("Second file: ih != nh : %d != %d\n",ih,nh+1);
    }
    fclose(fp2);
    /***********************************************************************/
    /* end reading the 2nd component (if mc >=2 )*/
    /***********************************************************************/
  }

  if (truedata >= 1){
    /***********************************************************************/
    /* reading "true" data for 1st component*/
    /***********************************************************************/
    fp1true = efopen(in1true, "r");
    if(!fgettra(fp1true,&tr,1)) err("for true data 1: can't read first trace of first file");

    /* Allocate memory for data */
    data1true   = ealloc2float(nt,nx1*nx2*nx3*nx4);
    /* Loop over traces */
    ih=0;
    memcpy((void *) data1true[ih],(const void *) tr.data,nt*sizeof(float));
    ih++;

    do {
      memcpy((void *) data1true[ih],(const void *) tr.data,nt*sizeof(float));
      ih++;
      if (ih > nh) err("Number of traces > %d\n",nh+1); 
    } while (fgettr(fp1true,&tr));
      fprintf(stderr,"done reading true data for 1st component: ih=%d\n",ih);

    fclose(fp1true);
    /***********************************************************************/
    /* end reading "true" data for 1st component*/
    /***********************************************************************/
    if (mc >= 2){
      /***********************************************************************/
      /* reading "true" data for 2nd component*/
      /***********************************************************************/
      fp2true = efopen(in2true, "r");
      if(!fgettra(fp2true,&tr,1)) err("for true data 2: can't read first trace of first file");
      
      /* Allocate memory for data*/ 
      data2true   = ealloc2float(nt,nx1*nx2*nx3*nx4);
      /* Loop over traces*/

      ih=0;
      memcpy((void *) data2true[ih],(const void *) tr.data,nt*sizeof(float));
      ih++;

      do {
      	memcpy((void *) data2true[ih],(const void *) tr.data,nt*sizeof(float));
      	ih++;
	if (ih > nh) err("Number of traces > %d\n",nh+1); 
      } while (fgettr(fp2true,&tr));
      fprintf(stderr,"done reading true data for 2nd component: ih=%d\n",ih);
      fclose(fp2true);
      /***********************************************************************/
      /* end reading "true" data for 2nd component*/
      /***********************************************************************/
    }
  }

  if (verbose) fprintf(stderr,"processing %d traces \n", nh);
  for (it=0;it<nt;it++) t[it]=t0+it*dt;  /* Not implemented for t0 != 0  */
  
  dx1=(float) (x1_max-x1_min)/(nx1-1);
  dx2=(float) (x2_max-x2_min)/(nx2-1);
  dx3=(float) (x3_max-x3_min)/(nx3-1);
  dx4=(float) (x4_max-x4_min)/(nx4-1);
	
  fprintf(stderr,"Using the following dimensions:\n"); 
  fprintf(stderr,"x1_min=%d, x1_max=%d, dx1=%f, nx1=%d\n",x1_min,x1_max,dx1,nx1); 
  fprintf(stderr,"x2_min=%d, x2_max=%d, dx2=%f, nx2=%d\n",x2_min,x2_max,dx2,nx2);
  fprintf(stderr,"x3_min=%d, x3_max=%d, dx3=%f, nx3=%d\n",x3_min,x3_max,dx3,nx3);
  fprintf(stderr,"x4_min=%d, x4_max=%d, dx4=%f, nx4=%d\n",x4_min,x4_max,dx4,nx4);

  for (ih=0;ih<nh;ih++){
    if (config==1){
      x1h[ih] = (int) roundf((x1[ih] - x1_min)/dx1);
      x2h[ih] = (int) roundf((x2[ih] - x2_min)/dx2);
      x3h[ih] = (int) roundf((x3[ih] - x3_min)/dx3);
      x4h[ih] = (int) roundf((x4[ih] - x4_min)/dx4);
      /* if any of the input data are outside of the limits then set them to the limits
         improvement to make: if the data fall outside then throw them away!*/
      if (x1h[ih] < 0) x1h[ih] = 0;
      if (x2h[ih] < 0) x2h[ih] = 0;
      if (x3h[ih] < 0) x3h[ih] = 0;
      if (x4h[ih] < 0) x4h[ih] = 0;
      if (x1h[ih] >= nx1) x1h[ih] = nx1 -1;
      if (x2h[ih] >= nx2) x2h[ih] = nx2 -1;
      if (x3h[ih] >= nx3) x3h[ih] = nx3 -1;
      if (x4h[ih] >= nx4) x4h[ih] = nx4 -1;
    }
    if (config==4){
      x1h[ih] = (int) x1[ih];
      x2h[ih] = (int) x2[ih];
      x3h[ih] = (int) x3[ih];
      x4h[ih] = (int) x4[ih];
    }
}
  
  x1hout = ealloc1int(nx1*nx2*nx3*nx4);
  x2hout = ealloc1int(nx1*nx2*nx3*nx4);
  x3hout = ealloc1int(nx1*nx2*nx3*nx4);
  x4hout = ealloc1int(nx1*nx2*nx3*nx4);
  x1out = ealloc1float(nx1*nx2*nx3*nx4);
  x2out = ealloc1float(nx1*nx2*nx3*nx4);
  x3out = ealloc1float(nx1*nx2*nx3*nx4);
  x4out = ealloc1float(nx1*nx2*nx3*nx4);

  ih = 0;
  for (ix1=0;ix1<nx1;ix1++){
    for (ix2=0;ix2<nx2;ix2++){
      for (ix3=0;ix3<nx3;ix3++){
	for (ix4=0;ix4<nx4;ix4++){
	  x1hout[ih] = ix1;
	  x2hout[ih] = ix2;
	  x3hout[ih] = ix3;
	  x4hout[ih] = ix4;
	  ih++;
	}
      }
    }
  }

  for (ih=0;ih<nx1*nx2*nx3*nx4;ih++){
    x1out[ih] = (float) (x1hout[ih])*dx1 + x1_min;
    x2out[ih] = (float) (x2hout[ih])*dx2 + x2_min;
    x3out[ih] = (float) (x3hout[ih])*dx3 + x3_min;
    x4out[ih] = (float) (x4hout[ih])*dx4 + x4_min;
  }

/***********************************************************************/
/* get the sampling operator*/
/***********************************************************************/
  if (dec){ /* additional plots only for debugging  */  
  /* (you can zero some traces to test interpolation) */
   rndnum = 0;
   for (ih=0;ih< nh;ih+=1){
     /* generate random number distributed between 0 and 1 */
     rndnum = franuni();
     if (rndnum<dec){
     for (it=0;it<nt;it++) data1in[ih][it]=0;
     }
   } 
  }	
	
  wd_no_pad  = ealloc1float(nh);

  for (ih=0; ih<nh;ih++){
    float sum = 0;
    for (it = 0; it< nt; it++) 
      sum += fabs(data1in[ih][it]);
    if (sum) wd_no_pad[ih]=1;
    else{ 
      wd_no_pad[ih]=0;
    }
  }

/***********************************************************************/
/* time windowing the reconstruction with sliding time windows*/
/***********************************************************************/
 twstart = 0;
 taper = 0;

 for (Itw=0;Itw<Ntw;Itw++){	
   if (Itw == 0){
	 Ntw = trunc(nt/(Ltw-Dtw));
	 if ( (float) nt/(Ltw-Dtw) - (float) Ntw > 0) Ntw++;
   }		

     twstart = Itw*(Ltw-Dtw);

     if (twstart+Ltw-1 >nt) twstart=nt-Ltw;
     

   if (Itw*(Ltw-Dtw+1) > nt){
      Ltw = Ltw + nt - Itw*(Ltw-Dtw+1);
   }
   for (ih=0;ih<nh;ih++){ 
     for (itw=0;itw<Ltw;itw++){
       data1in_tw[ih][itw] = data1in[ih][twstart+itw]*wd_no_pad[ih];
       if (mc >= 2) data2in_tw[ih][itw] = data2in[ih][twstart+itw]*wd_no_pad[ih];
     }
   }
   fprintf(stderr,"processing time window %d of %d\n",Itw+1,Ntw);

   if (mc == 1){
     process1c(data1in_tw,data1out_tw,verbose,Ltw,nh,dt,x1h,x2h,x3h,x4h,nx1,nx2,nx3,nx4,wd_no_pad,iter,iter_external,alphai,alphaf,bandi,bandf,fmax,method,ranki,rankf);
   }
   else if (mc == 2){
     process2c(data1in_tw,data2in_tw,data1out_tw,data2out_tw,verbose,Ltw,nh,dt,x1h,x2h,x3h,x4h,nx1,nx2,nx3,nx4,wd_no_pad,iter,iter_external,alphai,alphaf,bandi,bandf,fmax,method,ranki,rankf);
   }
   if (Itw==0){ 
     for (ih=0;ih<nx1*nx2*nx3*nx4;ih++){ 
       for (itw=0;itw<Ltw;itw++){   
	 data1out[ih][twstart+itw] = data1out_tw[ih][itw];
	 if (mc >= 2) data2out[ih][twstart+itw] = data2out_tw[ih][itw];
      }
     }	 	 
   }
   else{
     for (ih=0;ih<nx1*nx2*nx3*nx4;ih++){ 
       for (itw=0;itw<Dtw;itw++){   /* taper the top of the time window*/
	 taper = (float) ((Dtw-1) - itw)/(Dtw-1); 
	 data1out[ih][twstart+itw] = data1out[ih][twstart+itw]*(taper) + data1out_tw[ih][itw]*(1-taper);
	 if (mc >= 2) data2out[ih][twstart+itw] = data2out[ih][twstart+itw]*(taper) + data2out_tw[ih][itw]*(1-taper);
       }
       for (itw=Dtw;itw<Ltw;itw++){   
	 data1out[ih][twstart+itw] = data1out_tw[ih][itw];
	 if (mc >= 2) data2out[ih][twstart+itw] = data2out_tw[ih][itw];
       }
     }	 	 
   }
 }
 /***********************************************************************/
 /* end of processing time windows*/
 /***********************************************************************/


 /***********************************************************************/
 /* if true data was an auxillary output, then compare with reconstructed data*/
 /***********************************************************************/
 tmp_num1 = 0;
 tmp_denom1 = 0;
 Q1 = 0;
 tmp_num2 = 0;
 tmp_denom2 = 0;
 Q2 = 0;
 if (truedata >= 1){
   for (ih=0;ih<nx1*nx2*nx3*nx4;ih++){
     for (it = 1000; it< nt; it++){
       tmp_num1   += data1true[ih][it]*data1true[ih][it];
       tmp_denom1 += (data1out[ih][it] - data1true[ih][it])*(data1out[ih][it] - data1true[ih][it]);
       if (mc>=2){
	 tmp_num2   += data2true[ih][it]*data2true[ih][it];
	 tmp_denom2 += (data2out[ih][it] - data2true[ih][it])*(data2out[ih][it] - data2true[ih][it]);

       }
     }
   }
   Q1 = 10*log10f(tmp_num1/tmp_denom1);
   fprintf(stderr,"The quality of component 1 is %6.2f dB\n",Q1);
   if (mc >=2){
     Q2 = 10*log10f(tmp_num2/tmp_denom2);
     fprintf(stderr,"The quality of component 2 is %6.2f dB\n",Q2);
     }
 }

/***********************************************************************/
 /* re-calculate headers from their binned values*/
 deg2rad = PI/180;
 rad2deg = 180/PI;
 gammainv = 1/gamma;	 		

 imximy_out = ealloc1float(nx1*nx2*nx3*nx4);	    

 mx_rot = ealloc1float(nx1*nx2*nx3*nx4);	    
 my_rot = ealloc1float(nx1*nx2*nx3*nx4);	    
 mx_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 my_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 h_out  = ealloc1float(nx1*nx2*nx3*nx4);	    
 az_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 hx_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 hy_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 sx_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 sy_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 gx_out = ealloc1float(nx1*nx2*nx3*nx4);	    
 gy_out = ealloc1float(nx1*nx2*nx3*nx4);	    

 if (ang > 90) ang2=-deg2rad*(ang-90);
 else ang2=deg2rad*(90-ang);

 for (ih=0;ih<nx1*nx2*nx3*nx4;ih++){
   if (config==1){
   h_out[ih]  = (float) (x3out[ih] - 0.5)*dh;
   az_out[ih] = (float) (x4out[ih] - 0.5)*daz;
   /* compute hx and hy from h and az*/
   if (recip>0){ /*azimuths only range from 0-179.999 due to reciprocity */
     if (az_out[ih] <= 90){
       hx_out[ih] = h_out[ih]*cos(deg2rad*az_out[ih]);
       hy_out[ih] = h_out[ih]*sin(deg2rad*az_out[ih]);
     }
     else {
       hx_out[ih] = -h_out[ih]*cos(PI-(deg2rad*az_out[ih]));
       hy_out[ih] = h_out[ih]*sin(PI-(deg2rad*az_out[ih]));
     }
   }
   else{
     if (az_out[ih] <= 90){
       hx_out[ih] = h_out[ih]*cos(deg2rad*az_out[ih]);
       hy_out[ih] = h_out[ih]*sin(deg2rad*az_out[ih]);
     }
     else if (az_out[ih] > 90 && az_out[ih] <= 180) {
       hx_out[ih] = -h_out[ih]*cos(PI-(deg2rad*az_out[ih]));
       hy_out[ih] = h_out[ih]*sin(PI-(deg2rad*az_out[ih]));
     }
     else if (az_out[ih] > 180 && az_out[ih] <= 270) {
       hx_out[ih] = -h_out[ih]*cos((deg2rad*az_out[ih])-PI);
       hy_out[ih] = -h_out[ih]*sin((deg2rad*az_out[ih])-PI);
     }
     else {
       hx_out[ih] = h_out[ih]*cos(2*PI-(deg2rad*az_out[ih]));
       hy_out[ih] = -h_out[ih]*sin(2*PI-(deg2rad*az_out[ih]));
     }
   }
   imximy_out[ih] = (float) x1out[ih]*10000 + x2out[ih]; /* make a header that is unique for every bin location in the survey*/
   mx_rot[ih] = (float) (x1out[ih] - 0.5)*dmx + omx;
   my_rot[ih] = (float) (x2out[ih] - 0.5)*dmy + omy;
   /* note: negative rotation by ang2 requires the form of the rotation matrix shown below:*/
   mx_out[ih] =  (mx_rot[ih]-omx)*cos(ang2) + (my_rot[ih]-omy)*sin(ang2) + omx;
   my_out[ih] =  -(mx_rot[ih]-omx)*sin(ang2) + (my_rot[ih]-omy)*cos(ang2) + omy;
   sx_out[ih] = mx_out[ih] - hx_out[ih]/(1 + gammainv);
   sy_out[ih] = my_out[ih] - hy_out[ih]/(1 + gammainv);
   gx_out[ih] = mx_out[ih] + hx_out[ih]*(1-(1/(1 + gammainv)));
   gy_out[ih] = my_out[ih] + hy_out[ih]*(1-(1/(1 + gammainv)));
   }
   
   if (config==4){
   h_out[ih]  = (float) (x3out[ih] - 0.5)*dh;
   az_out[ih] = (float) (x4out[ih] - 0.5)*daz;
   /* compute hx and hy from h and az*/
   if (recip>0){ /*azimuths only range from 0-179.999 due to reciprocity */
     if (az_out[ih] <= 90){
       hx_out[ih] = h_out[ih]*cos(deg2rad*az_out[ih]);
       hy_out[ih] = h_out[ih]*sin(deg2rad*az_out[ih]);
     }
     else {
       hx_out[ih] = -h_out[ih]*cos(PI-(deg2rad*az_out[ih]));
       hy_out[ih] = h_out[ih]*sin(PI-(deg2rad*az_out[ih]));
     }
   }
   else{
     if (az_out[ih] <= 90){
       hx_out[ih] = h_out[ih]*cos(deg2rad*az_out[ih]);
       hy_out[ih] = h_out[ih]*sin(deg2rad*az_out[ih]);
     }
     else if (az_out[ih] > 90 && az_out[ih] <= 180) {
       hx_out[ih] = -h_out[ih]*cos(PI-(deg2rad*az_out[ih]));
       hy_out[ih] = h_out[ih]*sin(PI-(deg2rad*az_out[ih]));
     }
     else if (az_out[ih] > 180 && az_out[ih] <= 270) {
       hx_out[ih] = -h_out[ih]*cos((deg2rad*az_out[ih])-PI);
       hy_out[ih] = -h_out[ih]*sin((deg2rad*az_out[ih])-PI);
     }
     else {
       hx_out[ih] = h_out[ih]*cos(2*PI-(deg2rad*az_out[ih]));
       hy_out[ih] = -h_out[ih]*sin(2*PI-(deg2rad*az_out[ih]));
     }
   }
   	 imximy_out[ih] = (float) x1out[ih]*10000 + x2out[ih]; /* make a header that is unique for every bin location in the survey*/
   	 sx_out[ih] = omx;
     sy_out[ih] = omy;
     gx_out[ih] = sx_out[ih] + hx_out[ih];
     gy_out[ih] = sy_out[ih] + hy_out[ih];
   
     mx_out[ih] = sx_out[ih] + hx_out[ih]/(1 + gammainv); 
     my_out[ih] = sy_out[ih] + hy_out[ih]/(1 + gammainv); 
     h_out[ih]  = sqrt(hx_out[ih]*hx_out[ih] + hy_out[ih]*hy_out[ih]);
     /* azimuth measured from source to receiver
        CC from East and ranges from 0 to 359.999 degrees*/
     az_out[ih] = rad2deg*atan2((gy_out[ih]-sy_out[ih]),(gx_out[ih]-sx_out[ih]));
     if (az_out[ih] < 0.) az_out[ih] += 360.0;
     if (recip > 0.){ 
   	   if (az_out[ih] > 179.999){ 
	     az_out[ih] = az_out[ih] - 180;
       } 
     }
   }
   
 }
/***********************************************************************/

 /***********************************************************************/
 /* outputting 1st component:*/
 /***********************************************************************/
 fp1out = efopen(out1,"w");
 for (ih=0;ih<nx1*nx2*nx3*nx4;ih++){ 
   memcpy((void *) tr.data,(const void *) data1out[ih],nt*sizeof(float));
   tr.ntr  = (int) nx1*nx2*nx3*nx4;
   tr.ep  = (int) imximy_out[ih];
   tr.cdp  = (int) x1out[ih];
   tr.cdpt = (int) x2out[ih];
   tr.swdep= (int) x3out[ih];
   tr.gwdep= (int) x4out[ih];
   tr.sx  = (int) sx_out[ih];
   tr.sy  = (int) sy_out[ih];
   tr.gx  = (int) gx_out[ih];
   tr.gy  = (int) gy_out[ih];
   tr.gelev  = (int) mx_out[ih];
   tr.selev  = (int) my_out[ih];
   tr.gdel   = (int) hx_out[ih];
   tr.sdel   = (int) hy_out[ih];
   tr.offset = (int) h_out[ih];
   tr.otrav  = (int) az_out[ih];
   if (Wd[ih]==0) tr.trid=2; /* dead trace */
   fputtr(fp1out,&tr);
 }
 fclose(fp1out);
 /***********************************************************************/
 /* end outputting 1st component*/
 /***********************************************************************/

 if (mc >= 2){
   /***********************************************************************/
   /* outputting 2nd component:*/
   /***********************************************************************/
   fp2out = efopen(out2,"w");
   for (ih=0;ih<nx1*nx2*nx3*nx4;ih++){ 
     memcpy((void *) tr.data,(const void *) data2out[ih],nt*sizeof(float));
     tr.ntr  = (int) nx1*nx2*nx3*nx4;
     tr.ep  = (int) imximy_out[ih];
     tr.cdp  = (int) x1out[ih];
     tr.cdpt = (int) x2out[ih];
     tr.swdep= (int) x3out[ih];
     tr.gwdep= (int) x4out[ih];
     tr.sx  = (int) sx_out[ih];
     tr.sy  = (int) sy_out[ih];
     tr.gx  = (int) gx_out[ih];
     tr.gy  = (int) gy_out[ih];
     tr.gelev  = (int) mx_out[ih];
     tr.selev  = (int) my_out[ih];
     tr.gdel   = (int) hx_out[ih];
     tr.sdel   = (int) hy_out[ih];
     tr.offset = (int) h_out[ih];
     tr.otrav  = (int) az_out[ih];
     if (Wd[ih]==0) tr.trid=2; /* dead trace */
     fputtr(fp2out,&tr);
   }
   fclose(fp2out);
   /***********************************************************************/
   /* end outputting 2nd component*/
   /***********************************************************************/
 }

 /******** End of output **********/
 
 finish=time(0);
 elapsed_time=difftime(finish,start);
 fprintf(stderr,"Total time required: %f \n", elapsed_time);
 
 return EXIT_SUCCESS;
}









