all:
	clang++ -Wno-deprecated-register -mavx512f -O3 kernel.cc
