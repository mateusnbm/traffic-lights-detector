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

int calculateOrientationYellow(myPoints &mypoint, Mat original) {
	
	Mat verticalYellowROI;
	Mat horizontalYellowROI;
	Mat verticalRedROI;
	Mat horizontalRedROI;

	extractROI(original, verticalRedROI,      mypoint, VERTICAL,   RED_FLAG);
	extractROI(original, horizontalRedROI,    mypoint, HORIZONTAL, RED_FLAG);
	extractROI(original, verticalYellowROI,   mypoint, VERTICAL,   YEL_FLAG);
	extractROI(original, horizontalYellowROI, mypoint, HORIZONTAL, YEL_FLAG);

	float verticalYellowRatio   = 
		calcHistogram(verticalYellowROI,   "Vertical");

	float horizontalYellowRatio = 
		calcHistogram(horizontalYellowROI, "Horizontal");

	float verticalRedRatio   = 
		calcHistogram(verticalRedROI,      "Vertical");

	float horizontalRedRatio = 
		calcHistogram(horizontalRedROI,    "Horizontal");

	// Find max ratio
	float maxRatio = verticalYellowRatio;
	if (horizontalYellowRatio > maxRatio) maxRatio = horizontalYellowRatio;
	if (verticalRedRatio      > maxRatio) maxRatio = verticalRedRatio;
	if (horizontalRedRatio    > maxRatio) maxRatio = horizontalRedRatio;

	//
	if ((verticalYellowRatio + horizontalYellowRatio + 
		 verticalRedRatio    + horizontalRedRatio) / 4 < HISTOGRAM_THRESHOLD) {
		mypoint.orientation = NO_ORIENTATION;
	}
	else if (maxRatio == verticalYellowRatio) {
		mypoint.orientation = VERTICAL;
		mypoint.boundingBoxP1 = mypoint.yellowVerticalBBP1;
		mypoint.boundingBoxP2 = mypoint.yellowVerticalBBP2;
		mypoint.color = YEL_FLAG;
	}
	else if (maxRatio == horizontalYellowRatio) {
		mypoint.orientation = HORIZONTAL;
		mypoint.boundingBoxP1 = mypoint.yellowHorizontalBBP1;
		mypoint.boundingBoxP2 = mypoint.yellowHorizontalBBP2;
		mypoint.color = YEL_FLAG;
	}
	else if (maxRatio == verticalRedRatio) {
		mypoint.orientation = VERTICAL;
		mypoint.boundingBoxP1 = mypoint.redVerticalBBP1;
		mypoint.boundingBoxP2 = mypoint.redVerticalBBP2;
		mypoint.color = RED_FLAG;
	}
	else if (maxRatio == horizontalRedRatio) {
		mypoint.orientation = HORIZONTAL;
		mypoint.boundingBoxP1 = mypoint.redHorizontalBBP1;
		mypoint.boundingBoxP2 = mypoint.redHorizontalBBP2;
		mypoint.color = RED_FLAG;
	}

	printf("yv: %f yh: %f rv: %f rh: %f max: %f",
				verticalYellowRatio, horizontalYellowRatio, 
				verticalRedRatio, horizontalRedRatio,
				maxRatio);

	return mypoint.orientation;
}
