//
//
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace cv;

/* 
	extractROI 

*/

void extractROI(const Mat input, Mat &ROI, myPoints &mypoint, int orientation, int colorFlag) {

// VERTICAL BOUNDING BOX --------------------------------------------------
if (orientation == VERTICAL) {

	int BB_SIZE = ROI_BB_SIZE_MULTIPLIER*mypoint.area;
	int X = mypoint.x;
	int Y = mypoint.y;
	Point verticalp1;
	Point verticalp2;

	if (colorFlag == GRE_FLAG) {
		verticalp1.x = X-0.5*BB_SIZE;
		verticalp1.y = Y-1.5*BB_SIZE;
		verticalp2.x = X+0.5*BB_SIZE;
		verticalp2.y = Y+0.5*BB_SIZE;

		mypoint.greenVerticalBBP1 = verticalp1;
		mypoint.greenVerticalBBP2 = verticalp2;
	}
	else if (colorFlag == RED_FLAG) {
		verticalp1.x = X-0.5*BB_SIZE;
		verticalp1.y = Y-0.5*BB_SIZE;
		verticalp2.x = X+0.5*BB_SIZE;
		verticalp2.y = Y+1.5*BB_SIZE;

		mypoint.redVerticalBBP1 = verticalp1;
		mypoint.redVerticalBBP2 = verticalp2;
	}
	else if (colorFlag == YEL_FLAG) {
		verticalp1.x = X-0.5*BB_SIZE;
		verticalp1.y = Y-1.0*BB_SIZE;
		verticalp2.x = X+0.5*BB_SIZE;
		verticalp2.y = Y+1.0*BB_SIZE;		

		mypoint.yellowVerticalBBP1 = verticalp1;
		mypoint.yellowVerticalBBP2 = verticalp2;
	}

	// Checks image boundaries for bounding box
	int imgWidth  = input.size().width;
	int imgHeight = input.size().height;
	if (verticalp1.x < 0.0) verticalp1.x = 0.0;
	if (verticalp1.y < 0.0) verticalp1.y = 0.0;
	if (verticalp2.x > imgWidth)  verticalp2.x = imgWidth;
	if (verticalp2.y > imgHeight) verticalp2.y = imgHeight;

	// Checks image boundaries for ROI exctraction
	int ROIWidth  = BB_SIZE;
	int ROIHeight = ROI_MULTIPLIER*BB_SIZE;

	if (verticalp1.x + ROIWidth  > imgWidth)
		ROIWidth  = imgWidth - verticalp1.x;
	if (verticalp1.y + ROIHeight > imgHeight)
		ROIHeight = imgHeight - verticalp1.y;
	Rect ROIverticalregion(verticalp1.x+1, verticalp1.y+1, ROIWidth-1, ROIHeight-1);

	// Crops
	Mat verticalROI = input(ROIverticalregion);

	// Converts to grayscale
	Mat verticalROIGray;
	cvtColor(verticalROI, verticalROIGray, CV_BGR2GRAY);

	#ifdef APPLY_SOBEL_ROI
		// Sobel y
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;

		Mat grad_y;
		Sobel(verticalROIGray, grad_y, ddepth, 0, 1, 3, scale, delta, 
			  BORDER_DEFAULT);
		convertScaleAbs(grad_y, verticalROIGray);
	#endif

	verticalROIGray.copyTo(ROI);

} // end of if (vertical)

else if (orientation == HORIZONTAL) {

	int BB_SIZE = ROI_BB_SIZE_MULTIPLIER*mypoint.area;
	int X = mypoint.x;
	int Y = mypoint.y;
	Point horizontalp1;
	Point horizontalp2;

	if (colorFlag == GRE_FLAG) {
		horizontalp1.x = X-1.5*BB_SIZE;
		horizontalp1.y = Y-0.5*BB_SIZE;
		horizontalp2.x = X+0.5*BB_SIZE;
		horizontalp2.y = Y+0.5*BB_SIZE;

		mypoint.greenHorizontalBBP1 = horizontalp1;
		mypoint.greenHorizontalBBP2 = horizontalp2;
	}
	else if (colorFlag == RED_FLAG) {
		horizontalp1.x = X-0.5*BB_SIZE;
		horizontalp1.y = Y-0.5*BB_SIZE;
		horizontalp2.x = X+1.5*BB_SIZE;
		horizontalp2.y = Y+0.5*BB_SIZE;

		mypoint.redHorizontalBBP1 = horizontalp1;
		mypoint.redHorizontalBBP2 = horizontalp2;		
	}
	else if (colorFlag == YEL_FLAG) {
		horizontalp1.x = X-1.0*BB_SIZE;
		horizontalp1.y = Y-0.5*BB_SIZE;
		horizontalp2.x = X+1.0*BB_SIZE;
		horizontalp2.y = Y+0.5*BB_SIZE;

		mypoint.yellowHorizontalBBP1 = horizontalp1;
		mypoint.yellowHorizontalBBP2 = horizontalp2;
	}

	// Checks image boundaries for bounding box
	int imgWidth  = input.size().width;
	int imgHeight = input.size().height;
	if (horizontalp1.x < 0.0) horizontalp1.x = 0.0;
	if (horizontalp1.y < 0.0) horizontalp1.y = 0.0;
	if (horizontalp2.x > imgWidth)  horizontalp2.x = imgWidth;
	if (horizontalp2.y > imgHeight) horizontalp2.y = imgHeight;

	int ROIWidth  = ROI_MULTIPLIER*BB_SIZE;
	int ROIHeight = BB_SIZE;

	// Checks image boundaries for ROI exctraction
	if (horizontalp1.x + ROIWidth  > imgWidth)
		ROIWidth  = imgWidth - horizontalp1.x;
	if (horizontalp1.y + ROIHeight > imgHeight)
		ROIHeight = imgHeight - horizontalp1.y;
	Rect ROIhorizontalregion(horizontalp1.x+1, horizontalp1.y+1, 
							 ROIWidth-1, ROIHeight-1);

	Mat horizontalROI = input(ROIhorizontalregion);

	// Converts to grayscale
	Mat horizontalROIGray;
	cvtColor(horizontalROI, horizontalROIGray, CV_BGR2GRAY);

	#ifdef APPLY_SOBEL_ROI
		// Sobel X
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;

		Mat grad_x;
		Sobel(horizontalROIGray, grad_x, ddepth, 1, 0, 3, scale, delta, 
			  BORDER_DEFAULT );
		convertScaleAbs(grad_x, horizontalROIGray);
	#endif

	horizontalROIGray.copyTo(ROI);

} // end of if (horizontal)


else { // neither Horizontal or Vertical
	Mat temp(0,0,0);
	temp.copyTo(ROI);
}

}
