.PHONY: all clean

all: kernel test
	clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 kernel.o test.o -o a.out

clean:
	rm *.o a.out

test: test.cc
	clang++ -Wno-deprecated-register -mavx512f -O3 -c test.cc

kernel: kernel.cc kernel.h
	clang++ -Wno-deprecated-register -mavx512f -O3 -c kernel.cc

