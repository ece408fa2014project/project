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
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "easypng/png.h"
#include "sequential/sequential.cpp"
#include "cuda/edge_detection.cu"

using namespace std;
using namespace cv;
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
//     //begin by parsing command line arguments
//     if(argc < 2 || argc > 4)
//     {
//         cout << "Usage: " << argv[0] << " <filename> flags" << endl;
//         cout << "flags:\n\t-v verbose mode\n\t-e edge detection only \n\t-m motion detection only" << endl;
//         return -1;
//     }
//     //TODO implement reading flags later.
//
//     string filename(argv[1]);
//
//     PNG * image = new PNG;
//     if(!image->readFromFile(filename)) {
//         cout << "Error: unable to read image." << endl;
//         return -1;
//     }
//     #ifdef SEQUENTIAL
//         sequential_algorithm(image);
//     #endif //SEQUENTIAL
//
//     #ifdef CUDA
//         cuda_edge_algorithm(image);
//     #endif
    Mat cur, prev1, prev2, prev3, output;
    Mat BGR_3[3];
    VideoCapture cap(0);
    vector<vector<Point> > contours;
    namedWindow("Display");
    cap.read(cur);
    cur.convertTo(cur, CV_32FC1);
    Size s = cur.size();
    int rows = s.height;
    int columns = s.width;
    Mat out;

    while(1)
    {
        cap.read(cur);
        output = cur.clone();
        split(cur,BGR_3);
        out = BGR_3[0].clone();

        //Call kernel
        do_edge_detection_cuda((float *)BGR_3[2],(float*)BGR_3[1],(float*)BGR_3[0],(float*)out,(float*)prev1,(float*)prev2,(float*)prev3,columns,rows);

        cur = out.clone();
        prev3 = prev2.clone();
        prev2 = prev1.clone();
        prev1 = cur.clone();

        imshow("Display",out);
        if(waitKey(30) >= 0) break;
    }

}
