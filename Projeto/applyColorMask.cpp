//
//
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace cv;

#include "applyThreshold.h"

void applyColorMask(Mat &output, const Mat imgGray, Mat red_hue_img){

	Mat red_filtered;
	int mask_size = 6;

	// Prepare mask
	applyThreshold(red_hue_img);

	Mat element = getStructuringElement(MORPH_ELLIPSE,
										Size(mask_size, mask_size),
										Point(-1,-1));

	// Dilation
	dilate(red_hue_img, red_hue_img, element);

	// Blur
	blur(red_hue_img, red_hue_img, Size(BLUR_SIZE,BLUR_SIZE), Point(-1,-1));

	// Apply mask
	imgGray.copyTo(red_filtered, red_hue_img);

	output = red_filtered;
    
}
