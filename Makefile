.PHONY: all clean

all: Kernel Test
	clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 Kernel.o Test.o -o a.out

clean:
	rm *.o *.out

Test: Test.cc
	clang++ -Wno-deprecated-register -mavx512f -O3 -c Test.cc

Kernel: Kernel.cc Kernel.h
	clang++ -Wno-deprecated-register -mavx512f -O3 -c Kernel.cc

