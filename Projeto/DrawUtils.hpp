//
//  DrawUtils.hpp
//  Projeto
//
//  Created by Mateus Nunes de B Magalhaes on 6/21/16.
//  Copyright © 2016 MateusNBM. All rights reserved.
//

#ifndef DrawUtils_hpp
#define DrawUtils_hpp

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "StringUtils.hpp"

using namespace std;
using namespace cv;

typedef enum color {
    ColorRed = 0,
    ColorYellow = 1,
    ColorGreen = 2
} Color;

typedef enum orientation {
    OrientationUnknow = 0,
    OrientationPotrait = 1,
    OrientationLandscape = 2,
} Orientation;

typedef struct semaphore {
    
    Color color;
    Orientation orientation;
    unsigned int x;
    
} Semaphore;

void drawSemaphoreHUD(Mat image, vector<Semaphore> semaphores);

#endif /* DrawUtils_hpp */
