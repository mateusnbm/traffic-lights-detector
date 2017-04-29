//
//
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace cv;

#include "applyColorMask.h"
#include "applyBlobDetection.h"

#define AFTER_RED_COLOR_FILTER_BLUR 3
#define AFTER_GREEN_COLOR_FILTER_BLUR 7

void prepareBlobDetection(Mat &img,
                          Mat &imgHSV,
                          Mat &imgGray,
                          Mat &originalPreview,
                          size_t file_idx,
                          vector<myPoints> *points ) {

	// Color segmentation:
	Mat l_pre_color_img, u_pre_color_img; // Pre-color range filtering
	Mat l_red_hue_rng, u_red_hue_rng;			// Red ranges
	Mat green_hue_rng;										// Green range
	Mat yellow_hue_rng;										// Yellow range
	Mat red_hue_img;											// Red image

	Mat red_filtered;
	Mat green_filtered;
	Mat yellow_filtered;
	
    // Detect red blobs.
    
    inRange(imgHSV, Scalar(RED_L_LOW, 100, 50),  Scalar(RED_L_HIGH, 255, 255), l_red_hue_rng);
	inRange(imgHSV, Scalar(RED_H_LOW, 100, 50), Scalar(RED_H_HIGH, 255, 255), u_red_hue_rng);
    
	addWeighted(l_red_hue_rng, 1.0, u_red_hue_rng, 1.0, 0.0, red_hue_img);
    
	applyColorMask(red_filtered, imgGray, red_hue_img);
    
    blur(red_filtered, red_filtered, Size(AFTER_RED_COLOR_FILTER_BLUR, AFTER_RED_COLOR_FILTER_BLUR), Point(-1, -1));
    
	applyBlobDetection((int)file_idx, red_filtered, points, originalPreview, img, RED_FLAG);
	
    // Detect green blobs.
    
	inRange(imgHSV, Scalar(GRE_LOW, 100, 110),  Scalar(GRE_HIGH, 255, 255), green_hue_rng);
    
    applyColorMask(green_filtered, imgGray, green_hue_rng);
    
    blur(green_filtered, green_filtered, Size(AFTER_GREEN_COLOR_FILTER_BLUR, AFTER_GREEN_COLOR_FILTER_BLUR), Point(0, -1));
    
	applyBlobDetection((int)file_idx, green_filtered, points, originalPreview, img, GRE_FLAG);
	
    // Detect yellow blobs.
    
	inRange(imgHSV, Scalar(YEL_LOW, 100, 110), Scalar(YEL_HIGH, 255, 255), yellow_hue_rng);
    
	applyColorMask(yellow_filtered, imgGray, yellow_hue_rng);
    
	applyBlobDetection((int)file_idx, yellow_filtered, points, originalPreview, img, YEL_FLAG);
	
}
