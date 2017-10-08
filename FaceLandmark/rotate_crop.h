//
// Created by szhou on 3/14/16.
//

#ifndef FACELANDMARK_ROTATE_CROP_H
#define FACELANDMARK_ROTATE_CROP_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "imtransform.h"

using namespace cv;

Mat rc_from_path(const char* path, const int w, const int h, const double lx, const double ly, const double rx, const double ry);

Mat rc_from_mat(Mat src, const int w, const int h, const double lx, const double ly, const double rx, const double ry);

bool rc_from_path_to_path(const char* path, const int w, const int h, const double lx, const double ly, const double rx, const double ry, const char* dst);
#endif //FACELANDMARK_ROTATE_CROP_H
