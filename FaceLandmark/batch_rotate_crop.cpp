//
// Created by szhou on 3/14/16.
//

#include "batch_rotate_crop.h"
#include <boost/filesystem.hpp>
void swap(double &a, double &b) {
    double t;
    t = a;
    a = b;
    b = t;
}

int main (int argc, char** argv) {
    string src_list, root_path, dst_path;
    int img_width, img_height;
    if (argc != 6) {
        cout << "USAGE: " << argv[0] << " src_list width height root_path dst_path" << endl;
        return 1;
    } else {
        src_list = argv[1];
        root_path = argv[4];
        string temp_w = argv[2];
        string temp_h = argv[3];
        dst_path = argv[5];
        img_width = stoi(temp_w);
        img_height = stoi(temp_h);
    }
    cv::Size imgSize(img_width, img_width);

    ifstream src_file;
    src_file.open(argv[1], ios::in);
    if (!src_file.is_open()) {
        cout << "Open src file fail\n";
        return 2;
    }
    // namedWindow("Display", WINDOW_AUTOSIZE);
    string output_file = string("/data2/face_detection/samples/smile2.list");
    ofstream out_file;
    out_file.open(output_file);

    string img_path;
    string label;
    double lx, ly, rx, ry;
    int c = 0;
    while (!(src_file.eof())) {
        src_file >> img_path >> lx >> ly >> rx >> ry >> label;
        if (src_file.eof()) {
            break;
        }
//        if (rx < lx) {
//            swap(rx, lx);
//            swap(ry, ly);
//        }

        Mat img = imread(root_path + img_path);
        Mat transformed = sim_transform_image_3channels(img, img_width, img_height, lx, ly, rx, ry, 20, 38);
//        imshow("Display", transformed);
//        waitKey(0);
        string dst_image_name = dst_path + img_path;

        boost::filesystem::path path(dst_image_name);
        path.remove_filename();
        if (!boost::filesystem::exists(path))
            if (!boost::filesystem::create_directories(path)) {
                cerr << "ERROR CREATING DIR" << endl;
                return -1;
            }
        if (!imwrite(dst_image_name, transformed)) {
            cerr << "ERROR WRITING IMG" << endl;
            return -2;
        }
        out_file << img_path << " "  << label << endl;
        c++;
        cout << c << endl;
    }

    return 0;
}