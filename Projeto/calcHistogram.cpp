//
//
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace cv;

float calcHistogram(Mat ROIGray, String orientation) {

  Mat verticalHist;

  // Establish the number of bins
  int histSize = 210;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 210 } ;
  const float* histRange = { range };

  bool uniform = true; 
  bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &ROIGray, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; 
  int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  float verticalHistCount = 0;
  float verticalROIHistCount = 0;
  float verticalHistRatio = 0;
  int histMaxBin = 210;
  int histMaxROIBin = HISTOGRAM_ROI_MAX_BIN;

  /// Draw for each channel
  for( int j = 1; j < histSize; j++ ) {
  
  #ifdef SHOW_HISTOGRAM	
  	if (j < histMaxROIBin)
      line( histImage, Point( bin_w*(j-1), hist_h - cvRound(b_hist.at<float>(j-1)) ) ,
                       Point( bin_w*(j), hist_h - cvRound(b_hist.at<float>(j)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
    else
  		line( histImage, Point( bin_w*(j-1), hist_h - cvRound(b_hist.at<float>(j-1)) ) ,
                       Point( bin_w*(j), hist_h - cvRound(b_hist.at<float>(j)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
  #endif                 

    	verticalHistCount += cvRound(b_hist.at<float>(j));
    	if (j < histMaxROIBin) verticalROIHistCount += cvRound(b_hist.at<float>(j));

   }

  #ifdef SHOW_HISTOGRAM 
    // Display
    namedWindow(orientation, CV_WINDOW_AUTOSIZE);
    imshow(orientation, histImage);
  #endif

  //waitKey(0);

  return verticalROIHistCount / verticalHistCount;

}