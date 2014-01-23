/* Copyright (c) Signal Analysis and Imaging Group (SAIG), University of Alberta, 2013.*/
/* All rights reserved.                       */
/* sugeom  :  $Date: August     2012- Last version May 2013  */

#include "su.h"
#include "segy.h"
#include "header.h"
#include "cleansegy.h"
#include <time.h>

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

void save_gather(float **d, int nh, int nt, float dt, const char* name);
/*********************** self documentation **********************/
char *sdoc[] =  {
  " 	   							            ",
  " SUGEOM    set mx, my, hx, hy, h and az headers using shot and receiver  ",  
  "           coordinates. If binning=1 then outputs binned headers imx,    ",
  "           imy, ih and iaz using a grid defined by gridfile=gridfilename.",
  "           The grid file should have the following format:               ",
  "                ang                                                      ",
  "                omx omy                                                  ",
  "                dmx dmy                                                  ",
  "                dh  daz                                                  ",
  "                recip gamma                                              ",
  "           Where:                                                        ",
  " 	      ang= angle of receiver line orientation measured CC from      ",
  "                East in degrees (can range between 0-179.999)     	    ",
  "                program will orient binning as though rec-lines run N-S  ",
  " 	      omx= bin origin-x				                    ",
  " 	      omy= bin origin-y	                                            ",
  " 	      dmx= bin increment-x	                                    ",
  " 	      dmy= bin increment-y	                                    ",
  " 	      dh=  offset increment		                            ",
  " 	      daz= azimuth increment		                            ",
  " 	      recip= reciprocity flag (0=  no S-R recip=> az: 0:359.99)     ",
  "                                   (1= use S-R recip=> az: 0:179.99)     ",
  " 	      gamma=vp/vs (1 is default) used for CMP or ACP binning	    ",
  " 	      				                                    ",
  "           Because of the limited headers available in su, the following ",
  "           somewhat arbitrary choice of header word mapping is chosen:   ",
  " 	   				                                    ",
  " 	      su_header=its_meaning:	                                    ",
  " 	      gelev = mx				                    ",
  " 	      selev = my				                    ",
  " 	      gdel  = hx				                    ",
  " 	      sdel  = hy				                    ",
  " 	      offset= h				                            ",
  " 	      otrav = az				                    ",
  " 	      cdp   = imx			                            ",
  " 	      cdpt  = imy				                    ",
  " 	      swdep = ih				                    ",
  " 	      gwdep = iaz				                    ",
  " 	                                                                    ",
  "           nooutput=1 does not output any data                           ",
  "           maps=1 generates all of the maps described below              ",
  "           mapsg=1 plots the shot and receiver coords                    ",
  "           mapsg_rot=1 plots the shot and receiver coords after rotation ",
  "                     to 'ang' coordinate frame                           ",
  "           mapfold=1 plots the total fold                                ",
  "           maphxhy=1 plots offset-x vs offset-y                          ",
  "           maphaz=1 plots the offset vs the azimuth                      ",
  " 	   				                                    ",
  "           The program can also design a plan for 5d interpolation via   ",	    
  "           sliding blocks of bins. This option is activated by coding    ",
  "           planblocks=1. The user defines the block sizes by:            ",            
  "           bl_x=26				                            ",        
  "           bl_y=26				                            ",     
  "           And overlap lengths defined by                                ",
  "           tl_x=13                                                       ",
  "           tl_y=13                                                       ",
  "           For each block the program outputs two sets of files:         ",
  "           A text file with the dimensions of each block:                ",
  "           tmp_blocks/limits_block_blocknumber.txt                       ",
  "           which is formatted as follows:                                ",
  "                min_imx  max_imx  nimx                                   ",
  "                min_imy  max_imy  nimy                                   ",
  "                min_ih   max_ih   nih                                    ",
  "                min_iaz  max_iaz  niaz                                   ",
  "           A binary file of trace numbers for each trace in the block:   ",
  "           tmp_blocks/trnum_block_blocknumber.bin                        ",
  "                                                                         ",
  "           Make sure ntr is set. If it is not you can update it via      ",
  "           suchw < datain.su > dataout.su a=ntr_value b=1 key1=ntr       ",
 NULL};
/* Credits:
 * Aaron Stanton.
 * Trace header fields accessed: ntr, sx, sy, gx, gy
 * Last changes: May : 2013 
 */
/**************** end self doc ***********************************/

int main(int argc, char **argv)
{
  float *sx,*sy,*gx,*gy,*mx,*my,*hx,*hy,*h,*az,*mx_rot,*my_rot;
  int *imx,*imy,*ih,*iaz,*ihx,*ihy;
  float binning, recip, ang, ang2, omx, omy, dmx, dmy, dh, daz, dhx, dhy;
  float mapsg, mapsg_rot, mapfold, maphxhy, maphaz, maps, nooutput, planblocks;  
  int bl_x, bl_y, tl_x, tl_y;
  int verbose;
  segy tr; 
  time_t start,finish;
  double elapsed_time;	
  int ix, nx, nt, method;
  float gamma;	
  cwp_String gridfile;
  float **shots;
  float **recs;
  int i, j;
  float min_sx,max_sx,min_sy,max_sy,min_gx,max_gx,min_gy,max_gy;
  int NewLength;
  int nshots;
  float min_plot_x,min_plot_y,max_plot_x,max_plot_y,length_plot_x,length_plot_y,plot_length,plot_border;  
  float **shots_rot,**recs_rot;
  float rad2deg,deg2rad,gammainv;	 		
  char buf1[500],buf2[500],buf3[500],buf4[500],buf5[500];
  FILE *fp1,*fp2,*fp4,*fp5,*fp1_block,*fp2_block,*fp_grid;
  int nrecs;
  float min_sx_rot,min_sy_rot,min_gx_rot,min_gy_rot,min_plot_x_rot,min_plot_y_rot;
  int min_imx,max_imx,min_imy,max_imy;
  int nmx,nmy;
  float **fold;
  float max_hx,max_hy,max_h_plot,max_h,max_az,maxaz;
  int max_ih,max_iaz,max_ihx,max_ihy,num_imx,num_imy,max_nb_x,max_nb_y;
  float remainder_x,remainder_y;
  int *block_imx_min,*block_imy_min,*block_imx_max,*block_imy_max;
  int n_block,n,m;
  char buf_block_name1[50],buf_block_name2[50],buf_block_folder[50];
  float ix_float;
  
  fprintf(stderr,"*******SUGEOM*********\n");
  /* Initialize */
  initargs(argc, argv);
  requestdoc(1);

  start=time(0);    
  /* Get parameters */
  if (!getparfloat("binning", &binning)) binning = 0;
  if (!getparint("method", &method)) method = 0; /* (=0 use h az, =1 hx hy) */
  if (!getparfloat("mapsg", &mapsg)) mapsg = 0;
  if (!getparfloat("mapsg_rot", &mapsg_rot)) mapsg_rot = 0;
  if (!getparfloat("mapfold", &mapfold)) mapfold = 0;
  if (!getparfloat("maphxhy", &maphxhy)) maphxhy = 0;
  if (!getparfloat("maphaz", &maphaz)) maphaz = 0;
  if (!getparfloat("maps", &maps)) maps = 0;
  if (maps>0){
    mapsg = 1;
    mapsg_rot = 1;
    mapfold = 1;
    maphxhy = 1;
    maphaz = 1; 
  } 
  if (!getparfloat("nooutput", &nooutput)) nooutput = 0;
  if (binning>0){
    if (!getparstring("gridfile",&gridfile)) err("gridfile required."); 
    fp_grid=fopen(gridfile, "r");
    fscanf(fp_grid, "%f", &ang);
    fgetc(fp_grid);  /* go to the next line */
    fscanf(fp_grid, "%f %f",&omx,&omy);
    fgetc(fp_grid);  /* go to the next line */
    fscanf(fp_grid, "%f %f",&dmx,&dmy);
    fgetc(fp_grid);  /* go to the next line */
    fscanf(fp_grid, "%f %f",&dh,&daz);
    fgetc(fp_grid);  /* go to the next line */
    fscanf(fp_grid, "%f %f",&recip, &gamma);
    fclose(fp_grid);
    fprintf(stderr,"grid definition:\n ang=%f\n omx=%f, omy=%f\n dmx=%f, dmy=%f\n dh=%f, daz=%f\n recip=%f, gamma=%f\n", ang,omx,omy,dmx,dmy,dh,daz,recip,gamma);  
    if (ang<0 || ang >= 180) err("Make sure 0<=ang<180");
    if (recip==1) fprintf(stderr,"warning: using recip=1 (assuming reciprocity of sources and receivers)\n");
    if (!getparfloat("planblocks", &planblocks)) planblocks = 0;
  }
  if (planblocks>0){
    if (!getparint("bl_x", &bl_x)) bl_x = 26;
    if (!getparint("bl_y", &bl_y)) bl_y = 26;
    if (!getparint("tl_x", &tl_x)) tl_x = 13;
    if (!getparint("tl_y", &tl_y)) tl_y = 13;
  }
  if (!getparint("verbose", &verbose))  verbose =1;

  if (!gettr(&tr)) err("can't read first trace");
  if (!tr.ntr) err("ntr header field must be set");
  nx = (int) tr.ntr;
  nt = (int) tr.ns;
	
  sx=ealloc1float(nx);
  sy=ealloc1float(nx);
  gx=ealloc1float(nx);
  gy=ealloc1float(nx);
  mx=ealloc1float(nx);
  my=ealloc1float(nx);
  hx=ealloc1float(nx);
  hy=ealloc1float(nx);
  h =ealloc1float(nx);
  az=ealloc1float(nx);
  mx_rot=ealloc1float(nx);
  my_rot=ealloc1float(nx);
  imx=ealloc1int(nx);
  imy=ealloc1int(nx);
  ih =ealloc1int(nx);
  iaz=ealloc1int(nx);
  ihx=ealloc1int(nx);
  ihy=ealloc1int(nx);
	
  /* Loop over traces */
  /* Dead traces are marked with trid = 2. They are weighted to zero */
  ix=0;

  do {
    sx[ix]=(float) tr.sx;
    sy[ix]=(float) tr.sy;
    gx[ix]=(float) tr.gx;
    gy[ix]=(float) tr.gy;
    ix++;
    if (ix > nx) err("Number of traces > %d\n",nx); 
  } while (gettr(&tr));
  erewind(stdin);
  nx=ix;
  
  if (verbose) fprintf(stderr,"processing %d traces \n", nx);
  
  /*********************************************************************************/
  shots=ealloc2float(2,nx);
  recs=ealloc2float(2,nx);
  for (i=0;i<nx;i++){
    shots[i][0] = sx[i];
    shots[i][1] = sy[i];
    recs[i][0] =  gx[i];
    recs[i][1] =  gy[i];
  } 
  /*********************************************************************************/
  /* Remove duplicate shots to get only the coordinates of each shot once */
  /* new length of modified array */
  NewLength = 1;
  nshots = 0;
  for(i=0; i< nx; i++){
    for(j=0; j< NewLength ; j++){
      if((shots[i][0] == shots[j][0]) && (shots[i][1] == shots[j][1]))
  	break;
    }
    /* if none of the values in index[0..j] of array is not same as array[i],
       then copy the current value to corresponding new position in array */ 
    if (j==NewLength ){
      NewLength++;
      shots[NewLength][0] = shots[i][0];
      shots[NewLength][1] = shots[i][1];
      nshots++;
    }
  }
  if (nshots==0) nshots++; /*special case if only one shot */
  
   fprintf(stderr,"number of shots: %d\n", nshots);
  /*********************************************************************************/
  /*********************************************************************************/
  /* Remove duplicate receivers to get only the coordinates of each receiver once */
  /* new length of modified array */
  NewLength = 1;
  nrecs = 0;
  for(i=0; i< nx; i++){
    for(j=0; j< NewLength ; j++){
      if((recs[i][0] == recs[j][0]) && (recs[i][1] == recs[j][1]))
  	break;
    }
    /* if none of the values in index[0..j] of array is not same as array[i],
       then copy the current value to corresponding new position in array */
    if (j==NewLength){
      NewLength++;
      recs[NewLength][0] = recs[i][0];
      recs[NewLength][1] = recs[i][1];
      nrecs++;
    }
  }
  if (nrecs==0) nrecs++; /* special case if only one receiver */

  fprintf(stderr,"number of receivers: %d\n", nrecs);
  /*********************************************************************************/
  /* get min/max shot and receiver coordinates */
  min_sx = shots[0][0];
  max_sx = shots[0][0];
  min_sy = shots[0][1];
  max_sy = shots[0][1];
  min_gx = recs[0][0];
  max_gx = recs[0][0];
  min_gy = recs[0][1];
  max_gy = recs[0][1];
  for(i=0; i< nshots; i++){
    if (shots[i][0] < min_sx) min_sx = shots[i][0];
    if (shots[i][1] < min_sy) min_sy = shots[i][1];
    if (shots[i][0] > max_sx) max_sx = shots[i][0];
    if (shots[i][1] > max_sy) max_sy = shots[i][1];
  }
  for(i=0; i< nrecs; i++){
    if (recs[i][0] < min_gx) min_gx = recs[i][0];
    if (recs[i][1] < min_gy) min_gy = recs[i][1];
    if (recs[i][0] > max_gx) max_gx = recs[i][0];
    if (recs[i][1] > max_gy) max_gy = recs[i][1];
  }
  /*********************************************************************************/
    /* get plot dimensions */
    min_plot_x = min_sx;
    if (min_sx>min_gx) min_plot_x = min_gx;
    min_plot_y = min_sy;
    if (min_sy>min_gy) min_plot_y = min_gy;
    max_plot_x = max_sx;
    if (max_sx<max_gx) max_plot_x = max_gx;
    max_plot_y = max_sy;
    if (max_sy<max_gy) max_plot_y = max_gy;
    length_plot_x = max_plot_x - min_plot_x; 
    length_plot_y = max_plot_y - min_plot_y; 
    plot_length = length_plot_x;
    if (length_plot_x<length_plot_y) plot_length = length_plot_y;
    plot_border = 100;  

  /*********************************************************************************/
  if (mapsg>0){         

    fp1=fopen("tmp_sg.bin", "wb");
    for (ix=0; ix<nshots;ix++){
      fwrite(&shots[ix][0], sizeof(float), 1, fp1);
      fwrite(&shots[ix][1], sizeof(float), 1, fp1);
    }
    for (ix=0; ix<nrecs;ix++){
      fwrite(&recs[ix][0], sizeof(float), 1, fp1);
      fwrite(&recs[ix][1], sizeof(float), 1, fp1);
    }
    fclose(fp1);
    sprintf(buf1,"xgraph < tmp_sg.bin n=%d,%d nplot=2 x1beg=%f x1end=%f x2beg=%f x2end=%f linewidth=0,0 linecolor=2,4 mark=8,6 marksize=5,5 width=500 height=500 title='sources and receivers' windowtitle='sources and receivers' label1='x-coordinate (m)' label2='y-coordinate (m)' & \n",nshots,nrecs,min_plot_x - plot_border,min_plot_x + plot_length + plot_border,min_plot_y - plot_border,min_plot_y + plot_length + plot_border);
  system(buf1);
  }
  /*********************************************************************************/

  rad2deg = 180/PI;
  deg2rad = PI/180;
  gammainv = 1/gamma;	 		
  for (ix=0; ix<nx;ix++){
    hx[ix] = gx[ix] - sx[ix];
    hy[ix] = gy[ix] - sy[ix];
    h[ix]  = sqrt(hx[ix]*hx[ix] + hy[ix]*hy[ix]);
    /* azimuth measured from source to receiver
       CC from East and ranges from 0 to 359.999 degrees*/
    az[ix] = rad2deg*atan2((gy[ix]-sy[ix]),(gx[ix]-sx[ix]));
    if (az[ix] < 0.) az[ix] += 360.0;
    if (recip > 0.){ 
      if (az[ix] > 179.999){ 
	  az[ix] = az[ix] - 180;
      } 
    }
    mx[ix] = sx[ix] + hx[ix]/(1 + gammainv);
    my[ix] = sy[ix] + hy[ix]/(1 + gammainv);
  }

  if (ang > 90) ang2=-deg2rad*(ang-90);
  else ang2=deg2rad*(90-ang);

  dhx = dh; 
  dhy = daz;
  fprintf(stderr,"dhx=%f\n",dhx);
  fprintf(stderr,"dhy=%f\n",dhy);

  if (binning>0){
    for (ix=0; ix<nx;ix++){
      mx_rot[ix] =  (mx[ix]-omx)*cos(ang2) - (my[ix]-omy)*sin(ang2) + omx;
      my_rot[ix] =  (mx[ix]-omx)*sin(ang2) + (my[ix]-omy)*cos(ang2) + omy;
      imx[ix]    = (int) truncf(((mx_rot[ix])-omx)/dmx) + 1;
      imy[ix]    = (int) truncf(((my_rot[ix])-omy)/dmy) + 1;
      ih[ix]     = (int) truncf(h[ix]/dh) + 1;
      iaz[ix]    = (int) truncf(az[ix]/daz) + 1;
      ihx[ix]     = (int) truncf(hx[ix]/dhx) + 1;
      ihy[ix]     = (int) truncf(hy[ix]/dhy) + 1;
    }   
  }

  /*********************************************************************************/
  shots_rot=ealloc2float(2,nshots);
  recs_rot=ealloc2float(2,nrecs);
  if (mapsg_rot>0){
    if (ang==90) fprintf(stderr,"warning, ang=90 implies no rotation of coordinates.");
    for(i=0; i< nshots; i++){
      shots_rot[i][0] = (shots[i][0]-omx)*cos(ang2) - (shots[i][1]-omy)*sin(ang2) + omx;
      shots_rot[i][1] = (shots[i][0]-omx)*sin(ang2) + (shots[i][1]-omy)*cos(ang2) + omy;
    }
    for(i=0; i< nrecs; i++){
      recs_rot[i][0] = (recs[i][0]-omx)*cos(ang2) - (recs[i][1]-omy)*sin(ang2) + omx;
      recs_rot[i][1] = (recs[i][0]-omx)*sin(ang2) + (recs[i][1]-omy)*cos(ang2) + omy;
    }   


    /* get min/max shot and receiver coordinates after rotation of coordinates */
    min_sx_rot = shots_rot[0][0];
    min_sy_rot = shots_rot[0][1];
    min_gx_rot = recs_rot[0][0];
    min_gy_rot = recs_rot[0][1];
    for(i=0; i< nshots; i++){
      if (shots_rot[i][0] < min_sx_rot) min_sx_rot = shots_rot[i][0];
      if (shots_rot[i][1] < min_sy_rot) min_sy_rot = shots_rot[i][1];
    }
    for(i=0; i< nrecs; i++){
      if (recs_rot[i][0] < min_gx_rot) min_gx_rot = recs_rot[i][0];
      if (recs_rot[i][1] < min_gy_rot) min_gy_rot = recs_rot[i][1];
     }
    
    /* get plot dimensions */
    min_plot_x_rot = min_sx_rot;
    if (min_sx_rot>min_gx_rot) min_plot_x_rot = min_gx_rot;
    min_plot_y_rot = min_sy_rot;
    if (min_sy_rot>min_gy_rot) min_plot_y_rot = min_gy_rot;
    
    fp2=fopen("tmp_sg_rot.bin", "wb");
    for (ix=0; ix<nshots;ix++){
      fwrite(&shots_rot[ix][0], sizeof(float), 1, fp2);
      fwrite(&shots_rot[ix][1], sizeof(float), 1, fp2);
    }
    for (ix=0; ix<nrecs;ix++){
      fwrite(&recs_rot[ix][0], sizeof(float), 1, fp2);
      fwrite(&recs_rot[ix][1], sizeof(float), 1, fp2);
    }
    fclose(fp2);
    sprintf(buf2,"xgraph < tmp_sg_rot.bin n=%d,%d nplot=2 x1beg=%f x1end=%f x2beg=%f x2end=%f linewidth=0,0 linecolor=2,4 mark=8,6 marksize=5,5 width=500 height=500 title='sources and receivers after coordinate rotation' windowtitle='sources and receivers after coordinate rotation' label1='x-coordinate (m)' label2='y-coordinate (m)' & \n",nshots,nrecs,min_plot_x_rot - plot_border,min_plot_x_rot + plot_length + plot_border,min_plot_y_rot - plot_border,min_plot_y_rot + plot_length + plot_border);
  system(buf2);
  }
  /*********************************************************************************/
  /*********************************************************************************/
  /* get min/max imx/imy */
  min_imx = imx[0];
  max_imx = imx[0];
  min_imy = imy[0];
  max_imy = imy[0];
  for(i=0; i< nx; i++){
    if (imx[i] < min_imx) min_imx = imx[i];
    if (imx[i] > max_imx) max_imx = imx[i];
    if (imy[i] < min_imy) min_imy = imy[i];
    if (imy[i] > max_imy) max_imy = imy[i];
  }
  fprintf(stderr,"min/max inline number: %d %d\n",min_imx,max_imx);
  fprintf(stderr,"min/max crossline number: %d %d\n",min_imy,max_imy);
  nmx = max_imx-min_imx + 1;
  nmy = max_imy-min_imy + 1;
  fprintf(stderr,"number of inlines: %d\n",nmx);
  fprintf(stderr,"number of crosslines: %d\n",nmy);
  
  if (mapfold>0){         
    fold = ealloc2float(nmx+1,nmy+1);
    for(i=0; i< nmy; i++){
      for(j=0; j< nmx; j++){
	fold[i][j]=0;
      }
    }
      for(i=0; i< nx; i++){
	fold[imy[i]-min_imy][imx[i]-min_imx]++;
      }
   
      for(i=0; i< nmy; i++){
	for(j=0; j< nmx; j++){
	  /*      fprintf(stderr,"fold[%d][%d] = %f\n",i,j,fold[i][j]); */
	}
      }
    save_gather(fold,nmy,nmx,1,"tmp_fold.su");
    sprintf(buf3,"suximage < tmp_fold.su legend=1 cmap=hsv2 f1=%d d1=1 f2=%d d2=1 wbox=500 hbox=%d title='total fold' windowtitle='total fold' label1='imx' label2='imy' units='fold' perc=99 & \n",min_imx,min_imy,(int) 500*nmx/nmy);
    system(buf3);
  }
  /*********************************************************************************/

  /*********************************************************************************/
  if (maphxhy>0){ 
    /* get max hx/hy */
    max_hx = hx[0];
    max_hy = hy[0];
    for(i=0; i< nx; i++){
      if (abs(hx[i]) > max_hx) max_hx = abs(hx[i]);
      if (abs(hy[i]) > max_hy) max_hy = abs(hy[i]);
    }
    max_h_plot = max_hx;
    if (max_hy > max_h_plot) max_h_plot = max_hy;
    fp4=fopen("tmp_hxhy.bin", "wb");
    for (ix=0; ix<nx;ix++){
      fwrite(&hx[ix], sizeof(float), 1, fp4);
      fwrite(&hy[ix], sizeof(float), 1, fp4);
    }
    fclose(fp4);
    sprintf(buf4,"xgraph < tmp_hxhy.bin n=%d x1beg=%f x1end=%f x2beg=%f x2end=%f linewidth=0 linecolor=0 mark=8 marksize=5 width=500 height=500 title='offset-x vs. offset-y' windowtitle='offset-x vs. offset-y' label1='offset-x (m)' label2='offset-y (m)' & \n",nx,-max_h_plot - 50,max_h_plot + 50,-max_h_plot - 50,max_h_plot + 50);
    system(buf4);
  }
  /*********************************************************************************/
 
  /* get max h/az */
  max_h = h[0];
  max_az = az[0];
  for(i=0; i< nx; i++){
    if (abs(h[i]) > max_h) max_h = abs(h[i]);
    if (abs(az[i]) > max_az) max_az = abs(az[i]);
  }

  /*********************************************************************************/
  if (maphaz>0){ 
    fp5=fopen("tmp_haz.bin", "wb");
    for (ix=0; ix<nx;ix++){
      fwrite(&h[ix], sizeof(float), 1, fp5);
      fwrite(&az[ix], sizeof(float), 1, fp5);
    }
    maxaz = 360;
    if (recip > 0.) maxaz = 180;
    fclose(fp5);
    sprintf(buf5,"xgraph < tmp_haz.bin n=%d linewidth=0 linecolor=0 mark=8 marksize=5 width=500 height=500 x1beg=0 x1end=%f x2beg=0 x2end=%f title='offset vs. azimuth' windowtitle='offset vs. azimuth' label1='offset (m)' label2='azimuth (degrees, S->R CC from East)' & \n",nx,1.5*max_h,maxaz);
    system(buf5);
  }
  /*********************************************************************************/
   
  if (method==0){
  /* get max ih/iaz */
  max_ih = ih[0];
  max_iaz = iaz[0];
  for(i=0; i< nx; i++){
    if (ih[i] > max_ih) max_ih = ih[i];
    if (iaz[i] > max_iaz) max_iaz = iaz[i];
  }
  }
  else{
  /* get max ihx/ihy */
  max_ihx = ihx[0];
  max_ihy = ihy[0];
  for(i=0; i< nx; i++){
    if (ihx[i] > max_ihx) max_ihx = ihx[i];
    if (ihy[i] > max_ihy) max_ihy = ihy[i];
  }  
  }
  /*********************************************************************************/
  if (planblocks>0){
    /* design blocks of data for 5d reconstruction */
    
    num_imx = max_imx - min_imx + 1;
    num_imy = max_imy - min_imy + 1;

    max_nb_x = trunc((num_imx-tl_x)/(bl_x - tl_x));
    max_nb_y = trunc((num_imy-tl_y)/(bl_y - tl_y));

    remainder_x = (float) (num_imx-tl_x)/(bl_x - tl_x) - (float) max_nb_x;
    remainder_y = (float) (num_imy-tl_y)/(bl_y - tl_y) - (float) max_nb_y;
    if (remainder_x > 0) max_nb_x++;
    if (remainder_y > 0) max_nb_y++;

    block_imx_min = ealloc1int(nx);
    block_imy_min = ealloc1int(nx);
    block_imx_max = ealloc1int(nx);
    block_imy_max = ealloc1int(nx);

    n_block = 0;
    for (n=0; n<max_nb_x;n++){
      for (m=0; m<max_nb_y;m++){
  	block_imx_min[n_block] = (n)*bl_x - (n)*tl_x + min_imx;
  	block_imx_max[n_block] = block_imx_min[n_block] + bl_x - 1;
  	block_imy_min[n_block] = (m)*bl_y - (m)*tl_y + min_imy;
  	block_imy_max[n_block] = block_imy_min[n_block] + bl_y - 1;
	fprintf(stderr,"block[%d]: imx--> %d:%d   imy--> %d:%d\n",n_block,block_imx_min[n_block],block_imx_max[n_block],block_imy_min[n_block],block_imy_max[n_block]);
  	n_block++;
      }
    }
    if (method==0) fprintf(stderr,"there are %d blocks with dimensions (nmx*nmy*nh*naz*nt): %d*%d*%d*%d*%d \n",n_block,bl_x,bl_x,max_ih,max_iaz,nt);
    else fprintf(stderr,"there are %d blocks with dimensions (nmx*nmy*nh*naz*nt): %d*%d*%d*%d*%d \n",n_block,bl_x,bl_x,max_ihx,max_ihy,nt);
    
    /* now go through each block and see which trace numbers from the input file belong to it */
    sprintf(buf_block_folder,"mkdir tmp_blocks");
    system(buf_block_folder);
    for (n=0; n<n_block;n++){      
      sprintf(buf_block_name1,"tmp_blocks/trnum_block_%d.bin",n);
      fp1_block=fopen(buf_block_name1, "wb");
      for (ix=0; ix<nx;ix++){
	if ((imx[ix] >= block_imx_min[n]) && (imx[ix] <= block_imx_max[n]) && (imy[ix] >= block_imy_min[n]) && (imy[ix] <= block_imy_max[n])){
	  ix_float = (float) ix;
	  fwrite(&ix_float, sizeof(float), 1, fp1_block);
	}
      }
      fclose(fp1_block);
      sprintf(buf_block_name2,"tmp_blocks/limits_block_%d.txt",n);
      fp2_block=fopen(buf_block_name2, "w");
      fprintf(fp2_block, "%f %f %f\n",(float) block_imx_min[n],(float) block_imx_max[n],(float) bl_x);
      fprintf(fp2_block, "%f %f %f\n",(float) block_imy_min[n],(float) block_imy_max[n],(float) bl_y);
      if (method==0) fprintf(fp2_block, "%f %f %f\n",(float) 1,(float) max_ih,(float) max_ih );
	  else fprintf(fp2_block, "%f %f %f\n",(float) 1,(float) max_ihx,(float) max_ihx );
      if (method==0) fprintf(fp2_block, "%f %f %f\n",(float) 1,(float) max_iaz,(float) max_iaz );
	  else fprintf(fp2_block, "%f %f %f\n",(float) 1,(float) max_ihy,(float) max_ihy );
      fclose(fp2_block);
    }
  } 
  /*********************************************************************************/

  if (nooutput==0){
  rewind(stdin);
  for (ix=0;ix<nx;ix++){ 
    fgettr(stdin,&tr);
    tr.ntr=nx;
    tr.sx  = (int) sx[ix];
    tr.sy  = (int) sy[ix];
    tr.gx  = (int) gx[ix];
    tr.gy  = (int) gy[ix];
    tr.gelev  = (int) mx[ix];
    tr.selev  = (int) my[ix];
    tr.gdel   = (int) hx[ix];
    tr.sdel   = (int) hy[ix];
    tr.offset = (int) h[ix];
    tr.otrav  = (int) az[ix];
    if (binning>0){
      tr.cdp    = (int) imx[ix];
      tr.cdpt   = (int) imy[ix];
      if (method==0) tr.swdep  = (int) ih[ix];
      else tr.swdep  = (int) ihx[ix];
      if (method==0) tr.gwdep  = (int) iaz[ix];
      else tr.gwdep  = (int) ihy[ix];     
    }
    fputtr(stdout,&tr);
  }
  }
  /******** end of output **********/
 
  finish=time(0);
  elapsed_time=difftime(finish,start);
  fprintf(stderr,"Total time required: %f \n", elapsed_time);
 
  return EXIT_SUCCESS;
}

void save_gather(float **d, int nh, int nt, float dt, const char* name)
{

  int  itr;
  FILE* fp;
  segy trr;
  trr=cleansegy(trr);
 
  if ((fp=fopen(name,"w"))==NULL){ 
    warn("Cannot open fp\n");
    return;
  }
  for (itr=0;itr<nh;itr++){
      memcpy((void *) trr.data,
      (const void *) d[itr],nt*sizeof(float));
      trr.tracl=itr+1;
      trr.dt=(int) (dt*1e6);
      trr.ns=nt;
      trr.ntr=nh;
      fvputtr(fp,&trr);

  }    

  fclose(fp);  
  return;
  
}

