CXX=nvcc
LD=nvcc

EXE=security

CXX_FLAGS= -I/usr/local/cuda/include -lcudart -g -c -v
LD_INCLUDES=-L/usr/lib -L/usr/local/lib -I/usr/include -I/usr/local/include -I/usr/local/cuda/include -L/usr/local/cuda/lib64
LD_LIBS=-lpng -lc++ -lcudart
LD_FLAGS= -v -o $(EXE)

all: prep png util main
	$(LD) $(LD_INCLUDES) $(LD_LIBS) $(wildcard objs/*.o) $(LD_FLAGS)

all_stub:
	$(LD) $(LD_INCLUDES) $(LD_LIBS) $(wildcard objs/*.o) $(LD_FLAGS)

prep:
	rm -rf objs
	mkdir objs

main: main.cu
	echo "main"
	$(CXX) $(CXX_FLAGS) main.cu
	mv main.o objs

png: $(wildcard easypng/*.cpp)
	echo "png"
	$(CXX) $(CXX_FLAGS) $(wildcard easypng/*.cpp)
	mv *.o objs

sequential: $(wildcard sequential/*.cpp)
	echo "sequential"
	$(CXX) $(CXX_FLAGS) $(wildcard sequential/*.cpp)
	mv *.o objs

cuda:
	echo "cuda"
	$(CXX) $(CXX_FLAGS) $(wildcard cuda/*.cpp) $(wildcard cuda/*.cu)
	mv *.o objs

util: 
	echo "util"
	$(CXX) $(CXX_FLAGS) $(wildcard util/*.cpp)
	mv *.o objs

clean:
	rm -rf objs $(EXE)
