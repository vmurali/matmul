.PHONY: all clean

all: a.out

a.out: Kernel.o Test.o ThreadPool.o
	clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 Kernel.o Test.o ThreadPool.o -o a.out

clean:
	rm *.o *.out

Test.o: Test.cc Kernel.h ThreadPool.h
	clang++ -Wno-deprecated-register -mavx512f -O3 -c Test.cc

Kernel.o: Kernel.cc Kernel.h ThreadPool.h
	clang++ -Wno-deprecated-register -mavx512f -O3 -c Kernel.cc

ThreadPool.o: ThreadPool.cc ThreadPool.h
	clang++ -Wno-deprecated-register -mavx512f -O3 -c ThreadPool.cc
