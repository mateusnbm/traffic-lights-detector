//
//
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace cv;

#include "printUtils.h"
#include "calculateOrientation.h"
#include "calculateOrientationYellow.h"

size_t applyBlobDetection(int file_idx,
                          Mat &inputoutput,
                          vector<myPoints> *points,
                          Mat original,
                          Mat originalPreview,
                          int colorFlag){
    
    size_t i;
    
	vector<KeyPoint> keypoints;
    
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.5;
	params.filterByConvexity = true;
	params.minConvexity = 0.8;
	
    params.filterByColor = true;
    params.blobColor = 255;
    
	params.filterByCircularity = true;
    params.minCircularity = 0.7;
	params.filterByArea = true;
    params.minArea = 35; //35.0f;
	params.maxArea = 500.0f;
    
    SimpleBlobDetector blob_detector(params);
    blob_detector.detect(inputoutput, keypoints);
    
    drawKeypoints(originalPreview, keypoints, originalPreview,  Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    for(i = 0; i < keypoints.size(); i++){
        
		myPoints mypoint;
		mypoint.x = keypoints[i].pt.x;
		mypoint.y = keypoints[i].pt.y;
		mypoint.area = keypoints[i].size;
		mypoint.color = colorFlag;
		mypoint.frameId = file_idx;
        mypoint.roiHits = 1;
        
		// Calculate orientation and differenciate between RED and YELLOW
		if (colorFlag == GRE_FLAG || colorFlag == RED_FLAG) {
			mypoint.orientation =  calculateOrientation(mypoint, original, colorFlag);
		}else{
			mypoint.orientation =  calculateOrientationYellow(mypoint, original);
		}
        
		if(mypoint.color == GRE_FLAG) {
            mypoint.boundingBoxColor = Scalar(0, 255, 0);
        }else if(mypoint.color == YEL_FLAG) {
            mypoint.boundingBoxColor = Scalar(0, 255, 255);
        }else{
            mypoint.boundingBoxColor = Scalar(0, 0, 255);
        }
        
        rectangle(originalPreview, mypoint.boundingBoxP1, mypoint.boundingBoxP2, mypoint.boundingBoxColor);

		rectangle(originalPreview,
                  Point(mypoint.boundingBoxP1.x-1, mypoint.boundingBoxP1.y-1),
                  Point(mypoint.boundingBoxP2.x-1, mypoint.boundingBoxP2.y-1),
                  Scalar(0, 0, 0));

		mypoint.tracking_point = true;
        points->push_back(mypoint);
	
    }
    
	return keypoints.size();
    
}								
// ----------------------------------------------------------------------------


