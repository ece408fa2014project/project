CXX=nvcc
LD=nvcc

EXE=security

CXX_FLAGS= -g -c
LD_INCLUDES=-L/usr/lib -L/usr/local/lib -I/usr/include -I/usr/local/include
LD_LIBS=-lpng -lc++
LD_FLAGS= -o $(EXE)

all: prep png sequential cuda main
	$(LD) $(LD_INCLUDES) $(LD_LIBS) $(wildcard objs/*.o) $(LD_FLAGS)

prep:
	mkdir objs

main:
	$(CXX) $(CXX_FLAGS) main.cpp
	mv main.o objs

png:
	$(CXX) $(CXX_FLAGS) $(wildcard easypng/*.cpp)
	mv *.o objs

sequential:
	$(CXX) $(CXX_FLAGS) $(wildcard sequential/*.cpp)
	mv *.o objs

cuda:
	$(CXX) $(CXX_FLAGS) $(wildcard cuda/*.cpp)
	mv *.o objs

clean:
	rm -rf objs
