# SuperMUC only, intel mpi library
#CC = mpiicc
CC = mpic++

CFLAGS = -O3 -g -std=c++11 -fmessage-length=0 -Wno-unused-result -Wno-deprecated -pedantic -fopenmp

INCLUDES = -Isrc/ -Iinclude/ -Idep/SGpp/base/src/

LDFLAGS = -L/media/data/nfs/install/lib
LDFLAGS += -Llib -lsgppbase

SRC = src/main.cpp\
	  src/mpi/MPIObject.cpp\
      src/model/NS.cpp\
      src/surrogate/SGIDist.cpp\
      src/mcmc/MHMCMC.cpp\
      src/mcmc/PTMCMCDist.cpp\
      src/tools/io.cpp\
      src/sim/NSSim.cpp\
      src/sim/NSSimDist.cpp\
      src/sim/DomainDecomposer2D.cpp

DEP = src/config.h\
	  src/ForwardModel.hpp\
      src/mpi/MPIObject.hpp\
      src/model/NS.hpp\
      src/surrogate/SGIDist.hpp\
      src/mcmc/MHMCMC.hpp\
      src/mcmc/PTMCMCDist.hpp\
      src/tools/io.hpp\
      src/tools/matrix.hpp\
      src/sim/NSSim.hpp\
      src/sim/NSSimDist.hpp\
      src/sim/DomainDecomposer2D.hpp

OBJ = bin/main.o\
      bin/MPIObject.o\
      bin/NS.o\
      bin/SGIDist.o\
      bin/MHMCMC.o\
      bin/PTMCMCDist.o\
      bin/io.o\
      bin/NSSim.o\
      bin/NSSimDist.o\
      bin/DomainDecomposer2D.o

TARGET = bin/run

# Makefile variables meanings
# $@ : the target of the rule 
# $< : the 1st prerequisite of the rule
	
# create necessary directories (if not exist) before compiling
all:
	mkdir -p bin out; make run

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
run : $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) -lm $(LDFLAGS)

bin/main.o : src/main.cpp $(DEP)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ 

bin/MPIObject.o : src/mpi/MPIObject.cpp src/mpi/MPIObject.hpp src/config.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	
bin/NS.o : src/model/NS.cpp src/model/NS.hpp src/ForwardModel.hpp src/config.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

bin/SGIDist.o : src/surrogate/SGIDist.cpp src/surrogate/SGIDist.hpp src/ForwardModel.hpp src/config.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	
bin/MHMCMC.o : src/mcmc/MHMCMC.cpp src/mcmc/MHMCMC.hpp src/ForwardModel.hpp src/config.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	
bin/PTMCMCDist.o : src/mcmc/PTMCMCDist.cpp src/mcmc/PTMCMCDist.hpp src/config.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

bin/io.o : src/tools/io.cpp src/tools/io.hpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	
bin/NSSim.o : src/sim/NSSim.cpp src/sim/NSSim.hpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	
bin/NSSimDist.o : src/sim/NSSimDist.cpp src/sim/NSSimDist.hpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	
bin/DomainDecomposer2D.o : src/sim/DomainDecomposer2D.cpp src/sim/DomainDecomposer2D.hpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.PHONY: all sgpp clean clear

# Recompile SGpp library (needed for LRZ)
# Compile only the SGpp/base library
sgpp:
	mkdir -p lib; cd dep/SGpp/; scons -c; scons SG_ALL=0 SG_BASE=1 -j4; cd ../../

sgppclean:
	cd dep/SGpp; scons -c; cd ../../

clean:
	rm -rf bin/* out/*
	
clear:
	rm -rf out/*
