// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/opencv.hpp"
// #include <iostream>
//
// using namespace cv;
// using namespace std;
//
//
// int detectMotion(const Mat & motion, Mat & result, Mat & result_cropped, int x_start, int x_stop, int y_start, int y_stop, int max_deviation, Scalar & color)
// {
//     Scalar mean, stddev;
//     meanStdDev(motion, mean, stddev);
//
//     if(stddev[0] < max_deviation)
//     {
//       int num_changes = 0;
//       int min_x = motion.cols, max_x = 0;
//       int min_y = motion.rows, max_y = 0;
//
//       //Loop over image and detect changes
//       for(int j = y_start; j < y_stop; j+=2){
//           for(int i = x_start; i < x_stop; i+=2){
//               if(static_cast<int>(motion.at<uchar>(j,i)) == 255)
//               {
//                   num_changes++;
//                   if(min_x > i) min_x = i;
//                   if(max_x < i) max_x = i;
//                   if(min_y > j) min_y = j;
//                   if(max_y < j) max_y = j;
//               }
//           }
//       }
//
//       if(num_changes){
//         if(min_x-10 > 0) min_x -= 10;
//         if(min_y-10 > 0) min_y -= 10;
//         if(max_x+10 < result.cols-1) max_x += 10;
//         if(max_y+10 < result.rows-1) max_y += 10;
//
//         //Draw rectangle around the changed pixel
//         Point x(min_x,min_y);
//         Point y(max_x,max_y);
//         Rect rect(x,y);
//         Mat cropped = result(rect);
//         cropped.copyTo(result_cropped);
//         vector<vector<Point> > contours;
//         findContours(motion,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
//         drawContours(result,contours,-1,Scalar(0,0,255),2);
//         //rectangle(result,rect,color,1);
//       }
//
//       return num_changes;
//
//     }
//
//     return 0;
// }
//
// int main(int argc, char* argv[])
// {
//     VideoCapture cap(0); // open the video camera no. 0
//     cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//     cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
//
//     if (!cap.isOpened())  // if not success, exit program
//     {
//         cout << "Cannot open the video cam" << endl;
//         return -1;
//     }
//
//     double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
//     double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
//
//     cout << "Frame size : " << dWidth << " x " << dHeight << endl;
//
//     namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
//
//     // Take images and convert them to gray
//     Mat result, result_cropped, prev_frame, current_frame, next_frame;
//     cap.read(prev_frame);
//     result = prev_frame;
//     cap.read(current_frame);
//     cap.read(next_frame);
//
//     cvtColor(current_frame, current_frame, CV_RGB2GRAY);
//     cvtColor(prev_frame, prev_frame, CV_RGB2GRAY);
//     cvtColor(next_frame, next_frame, CV_RGB2GRAY);
//
//     /*Use d1 and d2 for calculating the differences
//     result: result of and operation - used on d1 and d2
//     num_changes: amount of changes in result matrix
//     color: color for drawing rectangle when something changes TODO*/
//
//     Mat d1, d2, motion;
//     int num_changes, num_sequence;
//     Scalar mean_, color(0,255,255); //yellow
//
//     //Detect motion in window
//     int x_start = 10, x_stop = current_frame.cols - 11;
//     int y_start = 350, y_stop = 530;
//
//     //Motion threshold
//     int motion_threshold = 5;
//
//     //Max deviation
//     int max_deviation = 20;
//
//     //Rectangle TODO
//     Mat kernel_ero = getStructuringElement(MORPH_RECT, Size(2,2));
//
//     //Now loop to detect motion
//     while (1)
//     {
//         //Retreive new image
//         prev_frame = current_frame;
//         current_frame = next_frame;
//         cap.read(next_frame);
//         result = next_frame;
//         cvtColor(next_frame, next_frame, CV_RGB2GRAY);
//
//         //Calculate difference between images and AND them
//         //TODO replace this with cuda kernel
//         absdiff(prev_frame, next_frame, d1);
//         absdiff(next_frame, current_frame, d2);
//         bitwise_and(d1, d2, motion);
//         threshold(motion, motion, 35, 255, CV_THRESH_BINARY);
//         erode(motion, motion, kernel_ero);
//
//         //Now detection Motion
//         //TODO will be replaced by CUDA kernel
//         num_changes = detectMotion(motion, result, result_cropped, x_start, x_stop, y_start, y_stop, max_deviation, color);
//
//         if(num_changes >= motion_threshold)
//         {
//             cout << "There is motion!!!" << endl;
//             cout << "filler" << endl;
//         }
//
//
//         Mat frame;
//
//         bool bSuccess = cap.read(frame); // read a new frame from video
//
//         if (!bSuccess) //if not success, break loop
//         {
//             cout << "Cannot read a frame from video stream" << endl;
//             break;
//         }
//
//         imshow("MyVideo", result); //show the frame in "MyVideo" window
//
//         if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
//         {
//             cout << "esc key is pressed by user" << endl;
//             break;
//         }
//
//     }
//
//     return 0;
//
// }

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
  Mat frame;
  Mat back;
  Mat fore;
  Mat BGR_3[3];
  VideoCapture cap(0);
  BackgroundSubtractorMOG2 bg(3,false);

  vector<vector<Point> > contours;

  namedWindow("Frame");

  while(1)
    {
      cap.read(frame);
      split(frame,BGR_3);

      /*for(int x = 0; x < 150; x++){
        for(int y = 0; y < 150; y++){
          frame.at<Vec3b>(y,x)[0] = 255;
          frame.at<Vec3b>(y,x)[1] = 255;
          frame.at<Vec3b>(y,x)[2] = 255;
        }
      }*/
      bg.operator ()(frame,fore);
      bg.getBackgroundImage(back);
      erode(fore,fore,Mat());
      dilate(fore,fore,Mat());
      findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
      drawContours(frame,contours,-1,Scalar(0,0,255),2);
      imshow("Frame",BGR_3[1]);
      if(waitKey(30) >= 0) break;
    }
    return 0;
  }
