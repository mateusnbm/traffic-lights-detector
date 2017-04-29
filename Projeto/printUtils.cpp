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
	printColorName()

*/
void printColorName(int colorFlag){
	switch(colorFlag) {
		case RED_FLAG: printf("RED"); break;
		case YEL_FLAG: printf("YELLOW"); break;
		case GRE_FLAG: printf("GREEN"); break;
	}
}

/*
		printOrientationName

*/
void printOrientationName(int orientationFlag){
	switch(orientationFlag) {
		case NO_ORIENTATION: printf("No-orientation"); break;
		case VERTICAL:			 printf("VERTICAL");  		 break;
		case HORIZONTAL:     printf("HORIZONTAL");		 break;
	}
}

/*
	printProjectVersion

*/
void printProjectVersion(int number_of_files) {

	printf("\nTraffic Lights Detection and Recognition\nVersion ");
	printf(VERSION_ID);
	printf("\n\n");
	printf("%d files loaded from directory ", number_of_files);
	printf(CURRENT_DATASET);
	printf("\n\n");

}



/*
	printResults

*/
void printResults(vector<myPoints> points) {

	printf("End of program. %d blobs found in total\n\n", (int)points.size());

	printf("Blob data: \n\n");
	for (unsigned i = 0; i < points.size(); i++) {
		printf("x: %f y: %f area: %f frame: %d color: ", points[i].x, points[i].y, 
				   points[i].area, points[i].frameId);
		printColorName(points[i].color);
		printf(" orientation: ");
		printOrientationName(points[i].orientation);
		printf("\n");
	}
	printf("\n\n");
	
}