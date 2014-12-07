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
#include "cuda/face_detection.cu"

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
    Mat cur, output;
    VideoCapture cap(0);
    vector<vector<Point> > contours;
    namedWindow("Display");
    cap.read(cur);
    Size s = cur.size();
    int rows = s.height;
    int columns = s.width;
    int size = rows * columns;
    float * r = (float*)malloc(size*sizeof(float));
    float * g = (float*)malloc(size*sizeof(float));
    float * b = (float*)malloc(size*sizeof(float));
    float * out_r = (float*)malloc(size*sizeof(float));
    float * out_g = (float*)malloc(size*sizeof(float));
    float * out_b = (float*)malloc(size*sizeof(float));
    float * prev1 = (float*)malloc(size*sizeof(float));
    float * prev2 = (float*)malloc(size*sizeof(float));
    float * prev3 = (float*)malloc(size*sizeof(float));
    float * out = (float*)malloc(size*sizeof(float));

    while(1)
    {
        cap.read(cur);
        output = cur.clone();

        for(int y = 0; y < rows; y++){
            for(int x = 0; x < columns; x++){
                b[columns*y+x] = cur.at<Vec3b>(y,x)[0];
                g[columns*y+x] = cur.at<Vec3b>(y,x)[1];
                r[columns*y+x] = cur.at<Vec3b>(y,x)[2];
                out_b[columns*y+x] = cur.at<Vec3b>(y,x)[0];
                out_g[columns*y+x] = cur.at<Vec3b>(y,x)[1];
                out_r[columns*y+x] = cur.at<Vec3b>(y,x)[2];
            }
        }

        //Call kernel
        //do_edge_detection_cuda(r,g,b,out,prev1,prev2,prev3,columns,rows);
        do_face_detection_cuda(r,g,b,out_r,out_g,out_b,columns,rows);

        //Copy back into Mat
        for(int y = 0; y < rows; y++){
            for(int x = 0; x < columns; x++){
                prev3[columns*y+x] = prev2[columns*y+x];
                prev2[columns*y+x] = prev1[columns*y+x];
                prev1[columns*y+x] = out[columns*y+x];

                cur.at<Vec3b>(y,x)[0] = out[rows*y+x];
                cur.at<Vec3b>(y,x)[1] = out[rows*y+x];
                cur.at<Vec3b>(y,x)[2] = out[rows*y+x];
                output.at<Vec3b>(y,x)[0] = out_b[columns*y+x];
                output.at<Vec3b>(y,x)[1] = out_g[columns*y+x];
                output.at<Vec3b>(y,x)[2] = out_r[columns*y+x];
            }
        }

        imshow("Display",output);
        if(waitKey(30) >= 0) break;
    }

}
