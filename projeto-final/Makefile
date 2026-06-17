SHELL:=/bin/bash
CODE_FILE=perceptron

all: compile run clean

clean:
	rm -f *.out ;
	rm -f $(CODE_FILE) ;

compile:
	export NVCC_CCBIN=g++13 ; \
	nvcc $(CODE_FILE).cu -o $(CODE_FILE) ;

run: compile
	- ./$(CODE_FILE) ;
