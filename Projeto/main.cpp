//
//  main.cpp
//  Projeto
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "configs.h"

using namespace std;
using namespace cv;

#include "StringUtils.hpp"
#include "DrawUtils.hpp"
#include "printUtils.h"
#include "prepareFrame.h"
#include "applyThreshold.h"
#include "prepareBlobDectection.h"

#define IMAGE_HEIGHT_FACTOR 0.5

#define TRACKED_POINT_HIT_UPPER_LIMIT 20
#define TRACKED_POINT_DISTANCE_THRESHOLD 50
#define TRACKED_POINT_CLEAR_THRESHOLD 0

#define TRACKER_FRAME_LIMIT 3

#define SEMAPHORE_DRAW_THRESHOLD 10

Rect calculateROIforCandidatePoint(myPoints point, Size imageSize) {
    
    int x1, x2, y1, y2;
    double width, height;
    
    x1 = point.boundingBoxP1.x - 20;
    y1 = point.boundingBoxP1.y - 20;
    x2 = point.boundingBoxP2.x + 20;
    y2 = point.boundingBoxP2.y + 20;
    
    // Check if the ROI respects the image boundaries.
    x1 = MIN(x1, x2);
    x1 = (x1 < 0 ? 0 : x1);
    y1 = MIN(y1, y2);
    y1 = (y1 < 0 ? 0 : y1);
    width = abs(x1-x2);
    width = ( (x1+width) >=  imageSize.width ? (imageSize.width-x1) : width);
    height = abs(y1-y2);
    height = ( (y1+height) >=  imageSize.height ? (imageSize.height-y1) : height);
    
    return Rect(x1, y1, width, height);
    
}

void on_trackbar(int, void *) {}

int main(int argc, char* argv[]) {
    
    int trackerFrameLimit = TRACKER_FRAME_LIMIT;
    int visualAidsOn = true;
    
    vector<myPoints> keypoints;
    
    //
    
    String folder = CURRENT_DATASET;
    vector<String> filenames;
    size_t fileIndex;
    
    size_t blobIndex;
    
    myPoints candidatePoint, trackedPoint;
    double distance;
    bool match;
    
    //
    
    Point trackPoint;
    Rect trackRect;
    Mat trackImage;
    
    vector<myPoints> tempPoints;
    size_t cmpIndex, pointIndex, keypointIndex;
    
    std::string str;
    
    size_t trackerCount = 0;
    
    vector<myPoints> points, scanPoints;
    
    bool enableTracking = false;
    
    Mat image, img, imgHSV, imgGray, imgOutput;
    
    // Define opencv windows.
    namedWindow("Result", WINDOW_AUTOSIZE);
    
    // Create control panel.
    //namedWindow("Control Painel", WINDOW_AUTOSIZE);
    //createTrackbar("Tracker Refresh Rate", "Control Painel", &trackerFrameLimit, 10, on_trackbar);
    //createTrackbar("Visual Aids On", "Control Painel", &visualAidsOn, 1, on_trackbar);
    
    while (1) {
        
        // Retrieve file paths from supplied dataset directory.
        glob(folder, filenames);
        
        // Loop through file paths.
        for(fileIndex = 0; fileIndex < filenames.size(); fileIndex++) {
            
            str = "Handling: " + String(filenames[fileIndex]) + " atIndex:" + to_string(fileIndex);
            std::cout << str << std::endl;
            
            image = imread(filenames[fileIndex], CV_LOAD_IMAGE_COLOR);
            
            image.copyTo(imgOutput);
            
            if(!image.data) {
                
                std::cout << "Error: Image failed to load." << std::endl;
                
            }else{
                
                if (trackerCount >= trackerFrameLimit) {
                    trackerCount = 0;
                    enableTracking = false;
                }else{
                    trackerCount++;
                }
                
                //enableTracking = false;
                //visualAidsOn = false;
                
                if (enableTracking == true) {
                    // Analyse each ROI and compute candidate points inside it.
                    for (pointIndex = 0; pointIndex < points.size(); pointIndex++) {
                        
                        // Calculate ROI around candidate point.
                        candidatePoint = points[pointIndex];
                        trackRect = calculateROIforCandidatePoint(candidatePoint, image.size());
                        // Extract ROI from the original image.
                        image(trackRect).copyTo(trackImage);
                        // Filter noise and detect blobs.
                        prepareFrame(trackImage, img, imgGray, imgHSV, IMAGE_HEIGHT_FACTOR, false);
                        applyThreshold(imgGray, 1, 0);
                        prepareBlobDetection(img, imgHSV, imgGray, trackImage, fileIndex, &tempPoints);
                        
                        //
                        
                        for (keypointIndex = 0; keypointIndex < tempPoints.size(); keypointIndex++) {
                            circle(imgOutput,
                                   Point(tempPoints[keypointIndex].x + trackRect.x,
                                         tempPoints[keypointIndex].y + trackRect.y),
                                   tempPoints[keypointIndex].area,
                                   tempPoints[keypointIndex].boundingBoxColor,
                                   CV_FILLED);
                            rectangle(imgOutput,
                                      scanPoints[keypointIndex].boundingBoxP1,
                                      scanPoints[keypointIndex].boundingBoxP2,
                                      scanPoints[keypointIndex].boundingBoxColor);
                        }
                        
                        //
                        
                        // Analyse candidate points.
                        if (tempPoints.size() > 0) {
                            // Check if any candidate point matches the point that generated the ROI.
                            for (cmpIndex = 0, match = false; cmpIndex < tempPoints.size() && match == false; cmpIndex++) {
                                // Translate the tracked point coordinates to the main image.
                                trackedPoint = tempPoints[cmpIndex];
                                trackedPoint.x += trackRect.x;
                                trackedPoint.y += trackRect.y;
                                // Calculate distance between points.
                                distance = norm(Point(candidatePoint.x, candidatePoint.y) - Point(trackedPoint.x, trackedPoint.y));
                                if (distance < TRACKED_POINT_DISTANCE_THRESHOLD) {
                                    // Match found. Update point boundary box.
                                    match = true;
                                    points[pointIndex].x = trackedPoint.x;
                                    points[pointIndex].y = trackedPoint.y;
                                    points[pointIndex].area = trackedPoint.area;
                                    points[pointIndex].color = trackedPoint.color;
                                    points[pointIndex].orientation = trackedPoint.orientation;
                                    points[pointIndex].boundingBoxP1 = Point(trackRect.x + trackedPoint.boundingBoxP1.x,
                                                                             trackRect.y + trackedPoint.boundingBoxP1.y
                                                                             );
                                    points[pointIndex].boundingBoxP2 = Point(trackRect.x + trackedPoint.boundingBoxP2.x,
                                                                             trackRect.y + trackedPoint.boundingBoxP2.y
                                                                             );
                                    points[pointIndex].boundingBoxColor = trackedPoint.boundingBoxColor;
                                    // Increate point ROI hit count.
                                    points[pointIndex].roiHits += (points[pointIndex].roiHits < TRACKED_POINT_HIT_UPPER_LIMIT ? 1 : 0);
                                }
                            }
                            // No point tracked is close enough to the point that generated the ROI.
                            if (match == false) {
                                points[pointIndex].roiHits -= 1;
                            }
                            // The tracked points are no longer needed.
                            tempPoints.clear();
                        }else{
                            // No candidate point has been found.
                            points[pointIndex].roiHits -= 1;
                        }
                    }
                    
                    // Clear points with low hit count.
                    for (pointIndex = 0; pointIndex < points.size(); pointIndex++) {
                        if (points[pointIndex].roiHits < TRACKED_POINT_CLEAR_THRESHOLD) {
                            points.erase(points.begin()+pointIndex);
                        }
                    }
                    // Disable tracking if there are no points left.
                    if (points.size() == 0) {
                        enableTracking = false;
                    }
                }
                
                if (enableTracking == false) {
                    // Extract the region of interest, suavize using blur, increase brightness and blue color scale value.
                    // Return copies of the processed image in grayscale and HSV.
                    prepareFrame(image, img, imgGray, imgHSV, IMAGE_HEIGHT_FACTOR, true);
                    // Apply ddaptive threshold. Dilate, no erode.
                    applyThreshold(imgGray, 1, 0);
                    // Run blob detection.
                    prepareBlobDetection(img, imgHSV, imgGray, image, fileIndex, &scanPoints);
                    
                    //
                    
                    for (keypointIndex = 0; keypointIndex < scanPoints.size(); keypointIndex++) {
                        circle(imgOutput,
                               Point(scanPoints[keypointIndex].x, scanPoints[keypointIndex].y),
                               scanPoints[keypointIndex].area,
                               scanPoints[keypointIndex].boundingBoxColor,
                               CV_FILLED);
                        rectangle(imgOutput,
                                  scanPoints[keypointIndex].boundingBoxP1,
                                  scanPoints[keypointIndex].boundingBoxP2,
                                  scanPoints[keypointIndex].boundingBoxColor);
                    }
                    
                    //
                    
                    // Merge tracked points and new ones.
                    points.insert(points.end(), scanPoints.begin(), scanPoints.end());
                    scanPoints.clear();
                }
                
                if (points.size() > 0) {
                    for (blobIndex = 0; blobIndex < points.size(); blobIndex++) {
                        
                        // Merge points that are too close to each other.
                        for (cmpIndex = blobIndex+1; cmpIndex < points.size(); cmpIndex++) {
                            double distance = norm(Point(points[blobIndex].x, points[blobIndex].y)-Point(points[cmpIndex].x, points[cmpIndex].y));
                            if (distance < TRACKED_POINT_DISTANCE_THRESHOLD) {
                                points[blobIndex].roiHits += abs(points[cmpIndex].roiHits);
                                points[blobIndex].roiHits = (points[blobIndex].roiHits <= TRACKED_POINT_HIT_UPPER_LIMIT ?
                                                             points[blobIndex].roiHits : TRACKED_POINT_HIT_UPPER_LIMIT);
                                points.erase(points.begin()+cmpIndex);
                                cmpIndex--;
                            }
                        }
                        
                        // Calculate ROI around candidate point.
                        candidatePoint = points[blobIndex];
                        trackRect = calculateROIforCandidatePoint(candidatePoint, image.size());
                        if (trackRect.width > 0 && trackRect.height > 0) {
                            
                            if (visualAidsOn) {
                                char message[] = "00\0";
                                if (candidatePoint.roiHits < 0) {
                                    message[0] = '-';
                                    message[1] = '0' + (abs(candidatePoint.roiHits)%10);
                                }else if (candidatePoint.roiHits < 10) {
                                    message[0] = '0';
                                    message[1] = '0' + (candidatePoint.roiHits%10);
                                }else{
                                    message[0] = '0' + (candidatePoint.roiHits/10);
                                    message[1] = '0' + (candidatePoint.roiHits%10);
                                }
                                putText(imgOutput, message, cvPoint(trackRect.x-12, trackRect.y-10), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
                                rectangle(imgOutput, trackRect, Scalar(255, 255, 255));
                            }
                            
                        }
                        
                    }
                    
                    enableTracking = true;
                    
                }
                
                // User feedback.
                
                vector<Semaphore> semaphores;
                
                for (pointIndex = 0; pointIndex < points.size(); pointIndex++) {
                    if (points[pointIndex].roiHits >= SEMAPHORE_DRAW_THRESHOLD) {
                        Semaphore semaphore;
                        semaphore.x = points[pointIndex].x;
                        semaphore.color = (Color)points[pointIndex].color;
                        semaphore.orientation = (Orientation)points[pointIndex].orientation;
                        semaphores.push_back(semaphore);
                    }
                }
                
                drawSemaphoreHUD(imgOutput, semaphores);
                imshow("Result", imgOutput);
                waitKey(100);
                
            }
            
        }
        
        tempPoints.clear();
        scanPoints.clear();
        points.clear();
        keypoints.clear();
        
    }
    
    return 0;
}
