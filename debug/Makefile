IMPIPATH=/work_fast/hellenbr/workspace/ihpcins_2016stable


CC=$(IMPIPATH)/bin/mpicxx

CFLAGS=-std=c++11 
CFLAGS+=-Wno-deprecated
CFLAGS+=-I../src -I../dep -I../dep/sgpp-base-2.0.0/base/src -I$(IMPIPATH)/include

CFLAGS+=-DGLOBAL_SCENARIO=1
CFLAGS+=-DMPI_MASTER=0
CFLAGS+=-DMCMC_PRINT_PROGRESS=1
CFLAGS+=-DSGI_PRINT_TIMER=1
CFLAGS+=-DSGI_PRINT_RANKPROGRESS=1
CFLAGS+=-DSGI_PRINT_GRIDPOINTS=1

LDFLAGS+=-L$(IMPIPATH)/lib -lmpi -lsgppbase

SRCDIR=../src
BUILDDIR=build_debug
TARGET=debug

DEP=$(SRCDIR)/tools/Config.hpp
DEP=$(SRCDIR)/tools/Parallel.hpp
DEP+=$(SRCDIR)/model/ForwardModel.hpp
DEP+=$(SRCDIR)/model/NS.hpp
DEP+=$(SRCDIR)/mcmc/MCMC.hpp
DEP+=$(SRCDIR)/mcmc/MetropolisHastings.hpp
DEP+=$(SRCDIR)/mcmc/ParallelTempering.hpp
DEP+=$(SRCDIR)/surrogate/SGI.hpp

OBJ=$(BUILDDIR)/debug.o
OBJ+=$(BUILDDIR)/Config.o
OBJ+=$(BUILDDIR)/Parallel.o
OBJ+=$(BUILDDIR)/NS.o
OBJ+=$(BUILDDIR)/MCMC.o
OBJ+=$(BUILDDIR)/MetropolisHastings.o
OBJ+=$(BUILDDIR)/ParallelTempering.o
OBJ+=$(BUILDDIR)/SGI.o


all: $(TARGET) 

$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ) $(LDFLAGS)

$(BUILDDIR)/debug.o: debug.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/Config.o: $(SRCDIR)/tools/Config.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/Parallel.o: $(SRCDIR)/tools/Parallel.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/NS.o: $(SRCDIR)/model/NS.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/MCMC.o: $(SRCDIR)/mcmc/MCMC.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/MetropolisHastings.o: $(SRCDIR)/mcmc/MetropolisHastings.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/ParallelTempering.o: $(SRCDIR)/mcmc/ParallelTempering.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/SGI.o: $(SRCDIR)/surrogate/SGI.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)


.PHONY: clean

clean:
	rm -rf $(BUILDDIR) $(TARGET)

dir:
	rm -rf $(BUILDDIR); mkdir -p $(BUILDDIR)
