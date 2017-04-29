//
//
//

#include "configs.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

void applyThreshold(Mat imgGray,
                    int enable_dilate = 1,
                    int enable_erode = 0,
					int blur_size = 2,
                    int morph_size = 2) {
    
    Mat kernel;
    
	blur(imgGray, imgGray, Size(blur_size, blur_size), Point(-1, -1));
    
    // Adaptive threshold using a histogram function.
	adaptiveThreshold(imgGray, imgGray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 35, -15);
    
	kernel = getStructuringElement(MORPH_ELLIPSE, Size(morph_size,morph_size), Point(-1,-1));
	
    if (enable_dilate) {
        dilate(imgGray, imgGray, kernel);
    }
    
    if (enable_erode) {
        erode(imgGray, imgGray, kernel);
    }
    
}
