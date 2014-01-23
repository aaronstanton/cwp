# Makefile for ...su/main

IDIR=-I$(CWPROOT)/include -I. 

include $(CWPROOT)/src/Makefile.config

D = $L/libcwp.a $L/libpar.a $L/libsu.a

LFLAGS= $(PRELFLAGS) -L$L -lsu -lpar -lcwp -lm -lfftw3f -llapack -lgfortran $(POSTLFLAGS)

PROGS =			\
	$B/suinterp5d	

INSTALL	:	$(PROGS)
	@-rm -f INSTALL
	@touch $@

$(PROGS):	$(CTARGET) $D 
	-$(CC) -std=c99 $(CFLAGS) $(IDIR) $(@F).c $(LFLAGS) -o $@
	@$(MCHMODLINE)
	@echo $(@F) installed in $B

remake	:
	-rm -f $(PROGS) INSTALL
	$(MAKE) 
	
clean::
	rm -f a.out junk* JUNK* core
