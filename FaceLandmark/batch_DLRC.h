//
// Created by szhou on 3/14/16.
//

#ifndef FACELANDMARK_FACE_DLRC_H
#define FACELANDMARK_FACE_DLRC_H

#include "stdafx.h"
#include "model.h"
#include "shapeRegression.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "NPDDetector.h"
#include "imtransform.h"



#include <string>
using std::string;

#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

#include <boost/uuid/uuid.hpp>
#include <boost/filesystem.hpp>
using boost::uuids::uuid;

#include <boost/uuid/uuid_generators.hpp>
using boost::uuids::random_generator;

#include <boost/uuid/uuid_io.hpp>

#include <time.h>

typedef struct face_info {
    string path;
    int p1_x;
    int p1_y;
    int p2_x;
    int p2_y;
} Face_info;

Point2d calc_eye_center(shapeXY shape, int left);

string formatted_output(bool status, int face_num, vector<Face_info> faces);

#endif //FACELANDMARK_FACE_DLRC_H
