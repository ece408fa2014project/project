/*
 * ECE 408 Final Project
 * Edge and motion detection in CUDA
 *
 * main.cpp
 * Author: Gabe Albacarys
 * Created: 12/02/14
 *
 * Main file for running the algo
 */

#include <iostream>
#include <string>
#include "easypng/png.h"
#include "sequential/sequential.cpp"
#include "cuda/cuda.cpp"

using namespace std;
/*
 * Inputs:
 *  COMMAND LINE ARGUMENTS
 *  Format: "edge-detector <filename> flags"
 *  Possible flags:
 *      -v verbose mode
 *      -e edge detection only, write edges back into png
 *      -m motion detection only, write motion file back into png
 *
 * Returns:
 *  -1 if error (i.e. file couldn't be loaded, CUDA errored out, etc.)
 *  0 if no motion is detected
 *  1 if motion is detected
 */
int main(int argc, char *argv[]) {
    //begin by parsing command line arguments
    if(argc < 2 || argc > 4)
    {
        cout << "Usage: " << argv[0] << " <filename> flags" << endl;
        cout << "flags:\n\t-v verbose mode\n\t-e edge detection only \n\t-m motion detection only" << endl;
        return -1;
    }
    //TODO implement reading flags later.

    string filename(argv[1]);

    PNG * image = new PNG;
    if(!image->readFromFile(filename)) {
        cout << "Error: unable to read image." << endl;
        return -1;
    }
    #ifdef SEQUENTIAL
        sequential_algorithm(image);
    #endif //SEQUENTIAL

    #ifdef CUDA
        cuda_algorithm(image);
    #endif
}
