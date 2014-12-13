#include "png.h"
#include "rgbapixel.h"
#include <cmath>
#include <ctime>
#include <sys/ioctl.h>

#define GRAYSCALE 1
#define GUASSIAN 1
#define SOBEL 1

// custom settings
#define INVERT 1
#define GFILTER 1

#define UPPERBOUND 100
#define LOWERBOUND 30
#define THRESHOLD 20

using namespace std;

int main()
{
	clock_t runtime = clock();
	// input png
	string inputname("lumetta.png");
	string outputname("lumettaout.png");
   	PNG * image = new PNG;
    if(!image->readFromFile(inputname)) {
        cout << "\nError: unable to read image." << endl;
        return -1;
    }
    int imageWidth = image->width();
    int imageHeight = image->height();

    PNG *out = new PNG(imageWidth,imageHeight);
	RGBAPixel *temp = new RGBAPixel;

clock_t algorithmtime = clock();
clock_t grayclock=0,gaussclock=0,sobelclock=0,edgeclock=0;

if(GRAYSCALE){
grayclock = clock();
	cout<<"\nComputing grayscale filter..."<<flush;
	//grayscale
	for(int ix=0;ix<imageWidth;ix++){
  		for(int iy=0;iy<imageHeight;iy++){
  			int color =0;
  			temp = (*image)(ix,iy);
  			color+= temp->red;
  			color+= temp->green;
  			color+= temp->blue;
  			color = color/3;
  			temp->red  =color;
  			temp->green=color;
  			temp->blue =color;
  			*(*out)(ix,iy) = *temp;
  		}
  	}
grayclock = clock()-grayclock;
  	cout<<"Done!"<<endl;
}

if(GUASSIAN){
gaussclock = clock();
	//int filterSum=159;int filter[5][5]={{2,4,5,4,2},{4,9,12,9,4},{5,12,15,12,5},{4,9,12,9,4},{2,4,5,4,2}};
  	int filterSum=273;int filter[5][5]={{1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,1},{4,16,26,16,4},{1,4,7,4,1}};

  	cout<<"Computing Guassian convolution..."<<flush;
  	for(int iy=2;iy<imageHeight-2;iy++){
  		for(int ix=2;ix<imageWidth-2;ix++){
  			int sumRed=0;
  			int sumGreen=0;
  			int sumBlue=0;
  			//load the surrounding pixels into convenient 3x3 pixel block
  			int irs[5][5] = {
  							{(*image)(ix-2,iy-2)->red, (*image)(ix-1,iy-2)->red, (*image)(ix,iy-2)->red, (*image)(ix+1,iy-2)->red, (*image)(ix+2,iy-2)->red},
  							{(*image)(ix-2,iy-1)->red, (*image)(ix-1,iy-1)->red, (*image)(ix,iy-1)->red, (*image)(ix+1,iy-1)->red, (*image)(ix+2,iy-1)->red},
  							{(*image)(ix-2,iy)->red,   (*image)(ix-1,iy)->red,   (*image)(ix,iy)->red,   (*image)(ix+1,iy)->red,   (*image)(ix+2,iy)->red},
  							{(*image)(ix-2,iy+1)->red, (*image)(ix-1,iy+1)->red, (*image)(ix,iy+1)->red, (*image)(ix+1,iy+1)->red, (*image)(ix+2,iy+1)->red},
  							{(*image)(ix-2,iy+2)->red, (*image)(ix-1,iy+2)->red, (*image)(ix,iy+2)->red, (*image)(ix+1,iy+2)->red, (*image)(ix+2,iy+2)->red}
  							};
  			int igs[5][5] = {
  							{(*image)(ix-2,iy-2)->green, (*image)(ix-1,iy-2)->green, (*image)(ix,iy-2)->green, (*image)(ix+1,iy-2)->green, (*image)(ix+2,iy-2)->green},
  							{(*image)(ix-2,iy-1)->green, (*image)(ix-1,iy-1)->green, (*image)(ix,iy-1)->green, (*image)(ix+1,iy-1)->green, (*image)(ix+2,iy-1)->green},
  							{(*image)(ix-2,iy)->green,   (*image)(ix-1,iy)->green,   (*image)(ix,iy)->green,   (*image)(ix+1,iy)->green,   (*image)(ix+2,iy)->green},
  							{(*image)(ix-2,iy+1)->green, (*image)(ix-1,iy+1)->green, (*image)(ix,iy+1)->green, (*image)(ix+1,iy+1)->green, (*image)(ix+2,iy+1)->green},
  							{(*image)(ix-2,iy+2)->green, (*image)(ix-1,iy+2)->green, (*image)(ix,iy+2)->green, (*image)(ix+1,iy+2)->green, (*image)(ix+2,iy+2)->green}
  							};
  			int ibs[5][5] = {
  							{(*image)(ix-2,iy-2)->blue, (*image)(ix-1,iy-2)->blue, (*image)(ix,iy-2)->blue, (*image)(ix+1,iy-2)->blue, (*image)(ix+2,iy-2)->blue},
  							{(*image)(ix-2,iy-1)->blue, (*image)(ix-1,iy-1)->blue, (*image)(ix,iy-1)->blue, (*image)(ix+1,iy-1)->blue, (*image)(ix+2,iy-1)->blue},
  							{(*image)(ix-2,iy)->blue,   (*image)(ix-1,iy)->blue,   (*image)(ix,iy)->blue,   (*image)(ix+1,iy)->blue,   (*image)(ix+2,iy)->blue},
  							{(*image)(ix-2,iy+1)->blue, (*image)(ix-1,iy+1)->blue, (*image)(ix,iy+1)->blue, (*image)(ix+1,iy+1)->blue, (*image)(ix+2,iy+1)->blue},
  							{(*image)(ix-2,iy+2)->blue, (*image)(ix-1,iy+2)->blue, (*image)(ix,iy+2)->blue, (*image)(ix+1,iy+2)->blue, (*image)(ix+2,iy+2)->blue}
  							};

			//convolve preloaded pixels with filter
			for(int sx=0;sx<5;sx++){
				for(int sy=0;sy<5;sy++){
					sumRed  +=irs[sy][sx]*filter[sy][sx];
					sumGreen+=igs[sy][sx]*filter[sy][sx];
					sumBlue +=ibs[sy][sx]*filter[sy][sx];
				}
			}
			//divide by filterSum
			temp->red = (sumRed/filterSum >255)?255:sumRed/filterSum;
			temp->green = (sumGreen/filterSum >255)?255:sumGreen/filterSum;
			temp->blue = (sumBlue/filterSum >255)?255:sumBlue/filterSum;

			//set pixel
			*(*out)(ix,iy) = *temp;
  		}
  	}
gaussclock = clock()-gaussclock;
  	cout<<"Done!"<<endl;
}

if(SOBEL){
sobelclock = clock();
  	int gx[3][3] = {{-1, 0, 1},
					{-2, 0, 2},
					{-1, 0, 1}};

	int gy[3][3] = {{ 1, 2, 1},
					{ 0, 0, 0},
					{-1,-2,-1}};

	//grad and edge dir
	int* gradients = new int[imageHeight*imageWidth];
	int* dirs = new int[imageHeight*imageWidth];

	cout<<"Computing Sobel convolution..."<<flush;
	//sobel convolution
	//int iw=0;
	//int ih=0;
	//if(imageWidth<imageHeight){iw=imageHeight; ih=imageWidth;}else{iw=imageWidth; ih=imageHeight;}

  	for(int iy=1;iy<imageHeight-1;iy++){
  	for(int ix=1;ix<imageWidth-1;ix++){
  		int sumx=0;
  		int sumy=0;
  		//load the surrounding pixels into convenient 3x3 pixel block
  		int irs[3][3] = {{(*image)(ix-1,iy-1)->red, (*image)(ix,iy-1)->red, (*image)(ix+1,iy-1)->red},
						 {(*image)(ix-1,iy  )->red, (*image)(ix,iy  )->red, (*image)(ix+1,iy  )->red},
						 {(*image)(ix-1,iy+1)->red, (*image)(ix,iy+1)->red, (*image)(ix+1,iy+1)->red}};

		//convolve preloaded pixels with gx,gy masks
		for(int sx=0;sx<3;sx++){
			for(int sy=0;sy<3;sy++){
				sumx+=irs[sy][sx]*gx[sy][sx];
				sumy+=irs[sy][sx]*gy[sy][sx];
			}
		}

		//take absolute value
		sumx=(sumx<0)?(sumx*-1):sumx;
		sumy=(sumy<0)?(sumy*-1):sumy;
		//add sumx+sumy, sumx is now total sum.
		sumx += sumy;
		//save your gradients!
		if(sumx<THRESHOLD){sumx=0;}
		gradients[iy*imageWidth+ix]=sumx;

		//calculate the angles
		float theta = atan2(sumy,sumx)/3.1415926 *180.0;
		int thetatwo;
		//Convert angle to approximate degree value
		if( ((theta < 22.5) && (theta > -22.5)) || (theta > 157.5) || (theta < -157.5)   ){thetatwo=  0;}
		if( ((theta > 22.5) && (theta <  67.5)) || ((theta <-112.5) && (theta > -157.5)) ){thetatwo= 45;}
		if( ((theta > 67.5) && (theta < 112.5)) || ((theta < -67.5) && (theta > -112.5)) ){thetatwo= 90;}
		if( ((theta >112.5) && (theta < 157.5)) || ((theta < -22.5) && (theta > -67.5))  ){thetatwo=135;}

		//save your directions!
		dirs[iy*imageWidth+ix]=thetatwo;

		if(sumx>255){sumx=255;}
		temp->red = 255-sumx;
		temp->green = 255-sumx;
		temp->blue = 255-sumx;
		if(INVERT){
			temp->red = sumx;
			temp->green = sumx;
			temp->blue = sumx;
		}
		*(*out)(ix,iy)=*temp;
  	}}//EOL
  	cout<<"Done!"<<endl;
sobelclock = clock()-sobelclock;
  	RGBAPixel *EDGE = new RGBAPixel(255,255,255,255);
  	RGBAPixel *NONE = new RGBAPixel(  0,  0,  0,255);

  	if(INVERT){
  		NONE->red=255;
  		NONE->green=255;
  		NONE->blue=255;
  		EDGE->red=0;
  		EDGE->green=0;
  		EDGE->blue=0;
  	}

  	cout<<"Computing edges..."<<flush;
edgeclock = clock();

  	for(int iy=1;iy<imageHeight-1;iy++){
    for(int ix=1;ix<imageWidth-1;ix++){

    	//gradient should be greater than UPPERBOUND
      if(gradients[iy*imageWidth+ix] > UPPERBOUND){
      	//currDir is the direction: 0,45,90,135
        int currDir = dirs[iy*imageWidth+ix];
        bool edgeFinished = false;
        int nextx=0,nexty=0;
        switch(currDir){
          case 0:
            nextx = ix+1;// check x in next direction
            nexty = iy;
            //Find and color next pixel in the same direction, within the bounds defined above
            while( !edgeFinished && (dirs[nexty*imageWidth+nextx]==currDir) && (gradients[nexty*imageWidth+nextx]>LOWERBOUND) ){
              *(*out)(ix,iy)=*EDGE;
              if(nextx>imageWidth-1){edgeFinished=true;}else{nextx+=1;} //increment x
            }
            break;
          case  45:
            nextx = ix+1;// check x in next direction
            nexty = iy+1;// check y in next direction
            while( !edgeFinished && (dirs[nexty*imageWidth+nextx]==currDir) && (gradients[nexty*imageWidth+nextx]>LOWERBOUND) ){
              *(*out)(ix,iy)=*EDGE;
              if(nextx>imageWidth-1){edgeFinished=true;}else{nextx+=1;}
              if(nexty>imageHeight-1){edgeFinished=true;}else{nexty+=1;}
            }
            break;
          case  90:
            nextx = ix;
            nexty = iy+1;// check y in next direction
            while( !edgeFinished && (dirs[nexty*imageWidth+nextx]==currDir) && (gradients[nexty*imageWidth+nextx]>LOWERBOUND) ){
              *(*out)(ix,iy)=*EDGE;
              if(nexty>imageHeight-1){edgeFinished=true;}else{nexty+=1;} //increment y
            }
            break;
          case 135:
            nextx = ix+1;//
            nexty = iy-1;// check y in next direction
            while( !edgeFinished && (dirs[nexty*imageWidth+nextx]==currDir) && (gradients[nexty*imageWidth+nextx]>LOWERBOUND) ){
              *(*out)(ix,iy)=*EDGE;
              if(nextx>imageWidth-1){edgeFinished=true;}else{nextx+=1;}
              if(nexty>imageHeight-1 || nexty < 0){edgeFinished=true;}else{nexty-=1;}
            }
            break;
          default : *(*out)(ix,iy)=*NONE;
        }
      }else{
        *(*out)(ix,iy)=*NONE;
      }
    }}//EOL

  	//cleanup non binary colors
  	for(int iy=1;iy<imageHeight-1;iy++){
  	for(int ix=1;ix<imageWidth-1;ix++){
  		if(*(*out)(ix,iy) != *EDGE ){
  			*(*out)(ix,iy)=*NONE;
  		}
  	}}//EOL
edgeclock = clock()-edgeclock;
algorithmtime = clock()-algorithmtime;

  	cout<<"Done!"<<endl;

  	delete EDGE;
  	delete NONE;
  	delete gradients;
  	delete dirs;
}

  	out->writeToFile(outputname);
  	delete image;
  	delete out;

  	runtime = clock()-runtime;

  	struct winsize w;
    ioctl(0, TIOCGWINSZ, &w);

  	cout<<"\nPerformance Results for: "<<imageWidth<<","<<imageHeight<<" px"<<endl;
  	for(int minux=0;minux<w.ws_col;minux++){
  		cout<<"-";
  	}
  	cout<<endl;
  	cout<<"Overall  : "<<runtime<<" clicks ("<<(float)runtime/CLOCKS_PER_SEC<<"s)"<<endl;
  	cout<<"Algorithm: "<<algorithmtime<<" clicks ("<<(float)algorithmtime/CLOCKS_PER_SEC<<"s)"<<endl;
  	cout<<"Grayscale: "<<grayclock<<" clicks ("<<(float)grayclock/CLOCKS_PER_SEC<<"s)"<<endl;
  	cout<<"Gaussian : "<<gaussclock<<" clicks ("<<(float)gaussclock/CLOCKS_PER_SEC<<"s)"<<endl;
  	cout<<"Sobel    : "<<sobelclock<<" clicks ("<<(float)sobelclock/CLOCKS_PER_SEC<<"s)"<<endl;
  	cout<<"Edges    : "<<edgeclock<<" clicks ("<<(float)edgeclock/CLOCKS_PER_SEC<<"s)"<<endl;
  	cout<<endl;
    return 0;
}
