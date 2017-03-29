BASEPATH = $(shell pwd)

all:
	scons

.PHONY: sgpp clean clear

sgpp:
	mkdir -p $(BASEPATH)/lib; cd $(BASEPATH)/dep/sgpp-base-2.0.0; scons -c; scons BUILDDIR=$(BASEPATH)/lib -j4; cd $(BASEPATH)
	
clean:
	scons -c