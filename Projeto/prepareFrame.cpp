//
//
//

#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace cv;

void prepareFrame(Mat &image,
                  Mat &img,
                  Mat &imgGray,
                  Mat &imgHSV,
                  double heightFactor,
                  bool crop) {
    
    Size imgSize;
    Rect upperROI;
    
    if (crop) {
        imgSize = image.size();
        // Assume the upper 1/3 of the image as the area of interest.
        upperROI = Rect(0, 0, imgSize.width, imgSize.height*heightFactor);
        // Extract the region of interest from the input image.
        image(upperROI).copyTo(img);
    }else{
        image.copyTo(img);
    }
    
    // Apply blur to the input image.
    blur(img, img, Size(BLUR_SIZE,BLUR_SIZE), Point(-1,-1));
    // Increase image brightness. Attenuate the blue color scale.
    img *= 1.2;
    img += Scalar(50, 0, 0);
    
    // Convert the processed image to grayscale and HSV.
    cvtColor(img, imgGray, CV_BGR2GRAY);
    cvtColor(img, imgHSV, CV_BGR2HSV);
    
}
