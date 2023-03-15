.PHONY: all clean

all: a.out

a.out: Kernel.o Test.o
	clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 Kernel.o Test.o -o a.out

clean:
	rm *.o *.out

Test.o: Test.cc Kernel.h
	clang++ -Wno-deprecated-register -mavx512f -O3 -c Test.cc

Kernel.o: Kernel.cc Kernel.h
	clang++ -Wno-deprecated-register -mavx512f -O3 -c Kernel.cc

