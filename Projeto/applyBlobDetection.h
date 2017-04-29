//
//
//

#pragma once

size_t applyBlobDetection(int file_idx,
                          Mat &inputoutput,
                          vector<myPoints> *points,
                          Mat original,
                          Mat originalPreview,
                          int colorFlag
                          );