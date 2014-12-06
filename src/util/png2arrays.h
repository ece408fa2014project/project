#ifndef __PNG2ARRAYS
#define __PNG2ARRAYS
//png2arrays.h
//
//Converts PNG's to an object with R, G, and B arrays

#include "../easypng/png.h"

class png2arrays {
    public:
        float* r;
        float* g;
        float* b;

        int x_dim;
        int y_dim;
        /*
         * Destructor.
         */
        ~png2arrays();

        /*
         * Parses the PNG and writes its RGB values into arrays
         */
        void parse_png(const PNG * image);

        /*
         * Since r, g, and b are on the heap (i.e. you access them)
         * via pointers, when you edit any of those values, you'll
         * be able to get a PNG out of them with this method
         */
        PNG * from_arrays();
};
#endif //__PNG2ARRAYS
