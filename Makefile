#GNU make

CXX=clang++
CXXFLAGS=-march=native -O3 -std=c++17
VPATH=src

obj=intel_sad.o

all:$(obj)

.PHONY:clean
clean:
	-rm -f $(obj)