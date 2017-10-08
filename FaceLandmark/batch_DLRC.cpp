//
// Created by szhou on 3/14/16.
//

#include "batch_DLRC.h"

Point2d calc_eye_center(shapeXY shape, int left)
{
    int start;
    if (left)
        start = 7;
    else
        start = 11;

    double x = 0, y = 0;
    for (int i = start; i < (start + 4); i++) {
        x += shape.shapeX[i];
        y += shape.shapeY[i];
    }
    x = x / 4;
    y = y / 4;
    return Point2d(x, y);
}

string make_uuid()
{
    return lexical_cast<string>((random_generator())());
}

string formatted_output(bool status, int face_num, vector<Face_info> faces) {
    string out = "";
    char line_1[100];
    snprintf(line_1, sizeof(line_1), "%d %d\n", status, face_num);
    out += line_1;
    for (int i = 0; i < face_num; i++) {
        out += faces[i].path + " " + to_string(faces[i].p1_x) + " " + to_string(faces[i].p1_y) + " " + to_string(faces[i].p2_x) + " " + to_string(faces[i].p2_y) + "\n";
    }
    return out;
}

int main(int argc, char** argv) {

    // Initialization
    string output_path = argv[3];
    string root_dir = argv[2];
    string list_file_path = argv[1];
    string face_D_Model = "/home/szhou/caffe_app/face_DLRC/model/npd_model_fro.dat";

    int minObjSize = 30;
    double scaleFactor = 1.2 ;
    double stepFactor = 0.05 ;

    CNPDDetector detector;
    int height = 250;
    int width = 250;
    if( !detector.InitDetector(face_D_Model.data(), height, width, minObjSize, scaleFactor, stepFactor) )
    {
//        cout << "Failed to initiate the NPD detector: " << modelFile << endl;
        cout << formatted_output(false, 0, vector<Face_info>());
        return -1;
    }

    ifstream in(list_file_path);
    string img_path;
    string full_path;
    while (in >> img_path) {
        full_path = root_dir + '/' + img_path;
        Mat img = imread(full_path, CV_LOAD_IMAGE_COLOR);
        detector.BGR2Gray(&img);
        vector<Rect> pResults;
        pResults.clear();
        detector.Detect(pResults);
        if (pResults.size() != 1) {
            continue;
        }
        vector<Face_info> faces;

        for (int i = 0; i < pResults.size(); i++) {
            _bbox bbox;

            bbox.rootcoord[0] = pResults[i].x;
            bbox.rootcoord[1] = pResults[i].y;
            bbox.rootcoord[2] = pResults[i].x + pResults[i].width;
            bbox.rootcoord[3] = pResults[i].y + pResults[i].height;

            // extract features start
            Model model("/home/szhou/caffe_app/face_DLRC/model/model_32pts_score.dat");
            shapeXY tShapeXY;
            shapeXY tShapeXY2;
            string shapeName;

            float score = 0.0;
            float roll, yaw, pitch;
            shapeRegression(model, img, bbox,tShapeXY2,score, roll, yaw, pitch);
            // printf("landmark location time: %dms\n", GetTickCount() - start);
            tShapeXY = tShapeXY2;

            int min_x = 9999.0;
            int min_y = 9999.0;
            int max_x = -9999.0;
            int max_y = -9999.0;
            for(int j=0; j<_pointNum; j++)
            {
                if(tShapeXY.shapeX[j]<min_x)
                    min_x = tShapeXY.shapeX[j];
                if(tShapeXY.shapeX[j]>max_x)
                    max_x = tShapeXY.shapeX[j];
                if(tShapeXY.shapeY[j]<min_y)
                    min_y = tShapeXY.shapeY[j];
                if(tShapeXY.shapeY[j]>max_y)
                    max_y = tShapeXY.shapeY[j];
            }
            // cout<<min_x<<" "<<min_y<<" "<<max_x<<" "<<max_y<<endl;
            int center_x = (min_x + max_x)/2;
            height = max_y - min_y;
            bbox.rootcoord[0] = center_x - height/2;
            bbox.rootcoord[1] = min_y;
            bbox.rootcoord[2] = center_x + height/2;
            bbox.rootcoord[3] = max_y;
            shapeRegression2(model, img, bbox,tShapeXY,tShapeXY2, score, roll, yaw, pitch);
//        Point2d left_eye_center = calc_eye_center(tShapeXY2, 1);
//        Point2d right_eye_center = calc_eye_center(tShapeXY2, 0);
//        Mat transformed = rc_from_mat(img, t_width, t_height, left_eye_center.x, left_eye_center.y, right_eye_center.x, right_eye_center.y);
            Point2d left_eye_center = calc_eye_center(tShapeXY2, 1);
            Point2d right_eye_center = calc_eye_center(tShapeXY2, 0);
            double ecx = (left_eye_center.x + right_eye_center.x) / 2;
            double ecy = (left_eye_center.y + right_eye_center.y) / 2;
            Point2d eye_center(ecx, ecy);
            Point2d mouth_center(tShapeXY2.shapeX[24], tShapeXY2.shapeY[24]);
            Mat transformed = sim_transform_image_3channels(img, 47, 55, eye_center.x, eye_center.y, mouth_center.x, mouth_center.y, 20, 38);
            string out_file_name = output_path + '/' + img_path;
            boost::filesystem::path path(out_file_name);
            path.remove_filename();
            if (!boost::filesystem::exists(path))
            if (!boost::filesystem::create_directories(path)) {
                cerr << "ERROR CREATING DIR" << endl;
                return -1;
            }
            imwrite(out_file_name, transformed);

            cout << img_path << endl;
        }
    }
    return 0;
}