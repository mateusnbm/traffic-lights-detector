//
//  DrawUtils.cpp
//  Projeto
//
//  Created by Mateus Nunes de B Magalhaes on 6/21/16.
//  Copyright © 2016 MateusNBM. All rights reserved.
//

#include "DrawUtils.hpp"

void drawSemaphoreHUD(Mat image, vector<Semaphore> semaphores) {
    
    size_t i;
    
    int viewHeight = (image.rows/3.0);
    int baseY = image.rows-viewHeight;
    
    int semaphoreWidth = 40;
    int semaphoreHeight = 100;
    Scalar semaphoreColor = Scalar(30, 30, 30);
    
    int lightRadius = 10;
    int lightOffset = 5;
    
    Scalar red, yellow, green;
    
    int x, y;
    
    // Draw background shape.
    rectangle(image, Rect(0, baseY, image.cols, viewHeight), Scalar(194, 192, 178), CV_FILLED);
    
    // Draw semaphores.
    
    for (i = 0; i < semaphores.size(); i++) {
        
        Semaphore semaphore = semaphores[i];
        
        switch (semaphore.color) {
            case ColorRed:
                red = Scalar(0, 0, 255);
                yellow = green = Scalar(0, 0, 0);
                break;
            case ColorYellow:
                yellow = Scalar(0, 255, 255);
                red = green = Scalar(0, 0, 0);
                break;
            case ColorGreen:
                green = Scalar(0, 255, 0);
                red = yellow = Scalar(0, 0, 0);
                break;
        }
        
        // Draw content based on orientation.
        if (semaphore.orientation == OrientationUnknow || semaphore.orientation == OrientationPotrait) {
            
            x = semaphore.x-(semaphoreWidth/2);
            y = baseY+((viewHeight-semaphoreHeight)/2);
            
            // Draw semaphore shape.
            rectangle(image, Rect(x, y, semaphoreWidth, semaphoreHeight), semaphoreColor, CV_FILLED);
            
            // Draw lights.
            x += semaphoreWidth/2;
            y += (semaphoreHeight-((lightRadius+lightOffset)*3)-lightOffset)/2;
            circle(image, Point(x, y), lightRadius, red, CV_FILLED);
            y += (lightRadius*2)+lightOffset;
            circle(image, Point(x, y), lightRadius, yellow, CV_FILLED);
            y += (lightRadius*2)+lightOffset;
            circle(image, Point(x, y), lightRadius, green, CV_FILLED);
            
        }else{
            
            x = semaphore.x-(semaphoreHeight/2);
            y = baseY+((viewHeight-semaphoreWidth)/2);
            
            // Draw semaphore shape.
            rectangle(image, Rect(x, y, semaphoreHeight, semaphoreWidth), semaphoreColor, CV_FILLED);
            
            // Draw lights.
            x += (semaphoreHeight/2)-(lightRadius*2)-lightOffset;
            y += semaphoreWidth/2;
            circle(image, Point(x, y), lightRadius, red, CV_FILLED);
            x += (lightRadius*2)+lightOffset;
            circle(image, Point(x, y), lightRadius, yellow, CV_FILLED);
            x += (lightRadius*2)+lightOffset;
            circle(image, Point(x, y), lightRadius, green, CV_FILLED);
            
        }
        
    }
    
}

void drawSemaphoreHUD(Mat image, Color highlightedColor, Orientation semaphoreOrientation, float confidence) {
    
    Scalar red, yellow, green;
    
    int x, y;
    
    int viewHeight = (image.rows/3.0);
    int baseY = image.rows-viewHeight;
    
    int semaphoreWidth = 40;
    int semaphoreHeight = 100;
    Scalar semaphoreColor = Scalar(30, 30, 30);
    
    int lightRadius = 10;
    int lightOffset = 5;
    
    int aprConfidence = floor(confidence);
    char message[] = "Confidence: 00%\0";
    char *confidenceStr = integerToChars(aprConfidence);
    message[12] = confidenceStr[0];
    message[13] = confidenceStr[1];
    
    // Define cicle colors based on the highlighted one.
    switch (highlightedColor) {
        case ColorRed:
            red = Scalar(0, 0, 255);
            yellow = green = Scalar(0, 0, 0);
            break;
        case ColorYellow:
            yellow = Scalar(0, 255, 255);
            red = green = Scalar(0, 0, 0);
            break;
        case ColorGreen:
            green = Scalar(0, 255, 0);
            red = yellow = Scalar(0, 0, 0);
            break;
    }
    
    // Draw background view.
    rectangle(image, Rect(0, baseY, image.cols, viewHeight), Scalar(194, 192, 178), CV_FILLED);
    
    // Draw confidence label on the left.
    putText(image, message, cvPoint(30, baseY+(viewHeight/2)+10), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 0), 1, CV_AA);
    
    // Draw content based on orientation.
    if (semaphoreOrientation == OrientationPotrait) {
        
        x = (image.cols-semaphoreWidth)/2;
        y = baseY+((viewHeight-semaphoreHeight)/2);
        
        // Draw semaphore shape.
        rectangle(image, Rect(x, y, semaphoreWidth, semaphoreHeight), semaphoreColor, CV_FILLED);
        
        // Draw lights.
        x += semaphoreWidth/2;
        y += (semaphoreHeight-((lightRadius+lightOffset)*3)-lightOffset)/2;
        circle(image, Point(x, y), lightRadius, red, CV_FILLED);
        y += (lightRadius*2)+lightOffset;
        circle(image, Point(x, y), lightRadius, yellow, CV_FILLED);
        y += (lightRadius*2)+lightOffset;
        circle(image, Point(x, y), lightRadius, green, CV_FILLED);
        
    }else{
        
        x = (image.cols-semaphoreHeight)/2;
        y = baseY+((viewHeight-semaphoreWidth)/2);
        
        // Draw semaphore shape.
        rectangle(image, Rect(x, y, semaphoreHeight, semaphoreWidth), semaphoreColor, CV_FILLED);
        
        // Draw lights.
        x += (semaphoreHeight/2)-(lightRadius*2)-lightOffset;
        y += semaphoreWidth/2;
        circle(image, Point(x, y), lightRadius, red, CV_FILLED);
        x += (lightRadius*2)+lightOffset;
        circle(image, Point(x, y), lightRadius, yellow, CV_FILLED);
        x += (lightRadius*2)+lightOffset;
        circle(image, Point(x, y), lightRadius, green, CV_FILLED);
        
    }
    
}