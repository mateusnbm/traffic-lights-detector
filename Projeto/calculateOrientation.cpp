//
//
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace cv;

#include "extractROI.h"
#include "calcHistogram.h"

//#include "applyThreshold.h"
//#include "applyColorMask.h"
//#include "printUtils.h"

int calculateOrientation(myPoints &mypoint, Mat original, int colorFlag) {
	
	Mat verticalROI;
	Mat horizontalROI;

	extractROI(original, verticalROI,    mypoint, VERTICAL,   colorFlag);
	extractROI(original, horizontalROI,  mypoint, HORIZONTAL, colorFlag);

	float verticalRatio   = calcHistogram(verticalROI,   "Vertical");
	float horizontalRatio = calcHistogram(horizontalROI, "Horizontal");

	if ((verticalRatio + horizontalRatio) / 2 < HISTOGRAM_THRESHOLD) {
		mypoint.orientation = NO_ORIENTATION;
	}
	else if (verticalRatio > horizontalRatio) {
		mypoint.orientation = VERTICAL;

		if (colorFlag == GRE_FLAG) {
			mypoint.boundingBoxP1 = mypoint.greenVerticalBBP1;
			mypoint.boundingBoxP2 = mypoint.greenVerticalBBP2;
		}
		else if (colorFlag == RED_FLAG) {
			mypoint.boundingBoxP1 = mypoint.redVerticalBBP1;
			mypoint.boundingBoxP2 = mypoint.redVerticalBBP2;
		}
		else if (colorFlag == YEL_FLAG) {
			mypoint.boundingBoxP1 = mypoint.yellowVerticalBBP1;
			mypoint.boundingBoxP2 = mypoint.yellowVerticalBBP2;
		}
	}
	else {
		mypoint.orientation = HORIZONTAL;

		if (colorFlag == GRE_FLAG) {
			mypoint.boundingBoxP1 = mypoint.greenHorizontalBBP1;
			mypoint.boundingBoxP2 = mypoint.greenHorizontalBBP2;
		}
		else if (colorFlag == RED_FLAG) {
			mypoint.boundingBoxP1 = mypoint.redHorizontalBBP1;
			mypoint.boundingBoxP2 = mypoint.redHorizontalBBP2;
		}
		else if (colorFlag == YEL_FLAG) {
			mypoint.boundingBoxP1 = mypoint.yellowHorizontalBBP1;
			mypoint.boundingBoxP2 = mypoint.yellowHorizontalBBP2;
		}
	}


	return mypoint.orientation;
}