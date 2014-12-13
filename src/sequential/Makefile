COMPILER = g++
COMPILER_FLAGS = -c -g -O0 -Wall
LINKER = g++

seqtest : motiondetect.o rgbapixel.o png.o 
	$(LINKER) motiondetect.o rgbapixel.o png.o -lpng -o seqtest

motiondetect.o : motiondetect.cpp rgbapixel.h png.h
	$(COMPILER) $(COMPILER_FLAGS) motiondetect.cpp

png.o : png.cpp png.h rgbapixel.h 
	$(COMPILER) $(COMPILER_FLAGS) png.cpp

rgbapixel.o : rgbapixel.cpp rgbapixel.h
	$(COMPILER) $(COMPILER_FLAGS) rgbapixel.cpp

clean: 
	-rm -f *.o seqtest
