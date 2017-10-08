//
// Created by szhou on 3/14/16.
//
#include "rotate_crop.h"
Mat rc_from_path(const char* path, const int w, const int h, const double lx, const double ly, const double rx, const double ry) {
    Mat src = imread(path);
    return rc_from_mat(src, w, h, lx, ly, rx ,ry);
}

bool rc_from_path_to_path(const char* path, const int w, const int h, const double lx, const double ly, const double rx, const double ry, const char* dst) {
    Mat src = imread(path);
    Mat cropped = rc_from_mat(src, w, h, lx, ly, rx ,ry);
    imwrite(dst, cropped);
    return true;
}

Mat rc_from_mat(Mat src, const int w, const int h, const double lx, const double ly, const double rx, const double ry) {
    Mat out = sim_transform_image_3channels(src, w, h, lx, ly, rx, ry);
    return out;
}