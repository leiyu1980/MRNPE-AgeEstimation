#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "imtransform.h"
#include <boost/filesystem.hpp>
#include <time.h>
#include <stdlib.h>
#include <strstream>
#include <math.h>

using namespace std;
using namespace cv;

extern float g_a;
extern float g_b;
extern float g_tx;
extern float g_ty;

float g_a = 0;
float g_b = 0;
float g_tx = 0;
float g_ty = 0;

bool stop = false;
void perturb(float &e_c_x, float &e_c_y, float &u_l_x, float &u_l_y, int max_offset) {
    srand (unsigned(time (NULL)));
    bool perturb_eye = rand() % 2 == 1;
    int offset_x = rand() % (max_offset * 2 + 1) - max_offset;
    int offset_y = rand() % (max_offset * 2 + 1) - max_offset;
    if (perturb_eye) {
        e_c_x += float(offset_x);
        e_c_y += float(offset_y);
    } else {
        u_l_x += float(offset_x);
        u_l_y += float(offset_y);
    }
}


int main( int argc, char** argv )
{

    // argv[1] list file
    // argv[2] image_src_parent_dir
    // argv[3] image_dst_parent_dirs
	ifstream in(argv[1]);
	string parent_dir_save = argv[3];
	string parent_dir = argv[2];
	string shapeName="";
	string image_name="";
    string _image_name ="";
    double tempx;
    double tempy;
    double distance19;
    double distance18;
	string tmpString;
	double tmp;
    cv::Mat img_transform2;
    int b;
	string::size_type sz;
    int count = 0;
    cout<< g_a<<endl;
    Point points[21];
    Point d_points[21];
	while (in>>image_name)
	{
        count++;
        b = 0;
        try {
            //cout<<count<<endl;
            distance18 = 0.0;
            distance19 = 0.0;
            image_name = image_name.replace(image_name.find(".shape"), 8, "");
            //cout<<image_name<<endl;
            Mat img = imread(parent_dir+image_name,CV_LOAD_IMAGE_COLOR);
            string tmp_str = image_name;
            b = tmp_str.find("jpg");
            //cout<<b<<endl;
            if(b!=-1)
                shapeName =  parent_dir + tmp_str.replace(tmp_str.find("jpg"), 3, "jpg.shape");
            else
                shapeName =  parent_dir + tmp_str.replace(tmp_str.find("JPG"), 3, "JPG.shape");
            //cout<<shapeName<<endl;
            Mat src_img;
            img.copyTo(src_img);


            ifstream inshape(shapeName);
            inshape >> tmpString;
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//1
            //cout<<tempx<<" "<<tempy<<endl;
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"0",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[0] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"1",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[1] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//3
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"2",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[2] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"3",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[3] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//5
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"4",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[4] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"5",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[5] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//7
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            putText(img,"6",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[6] = Point(tempx,tempy);
            double eye_center_x = tempx;//eye center
            double eye_center_y = tempy;//eye center
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            putText(img,"7",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[7] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//9
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            putText(img,"8",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[8] = Point(tempx,tempy);
            double up_lip_center_x = tempx;
            double up_lip_center_y = tempy;
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            putText(img,"9",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[9] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//11
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"10",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[10] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"11",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[11] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//13
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"12",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[12] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"13",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[13] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//15
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"14",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[14] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"15",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[15] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//17
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            putText(img,"16",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[16] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"17",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[17] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//19
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //cout<<tempx<<" "<<tempy<<endl;
            putText(img,"18",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[18] = Point(tempx,tempy);
            inshape >> tmpString;inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            putText(img,"19",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[19] = Point(tempx,tempy);
            inshape >> tmpString; inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);//20
            //circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            //putText(img,"20",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[20] = Point(tempx,tempy);
            inshape >> tmpString; inshape >> tmpString;tempx=stod(tmpString);inshape >> tmpString;tempy=stod(tmpString);
            circle(img,Point(tempx,tempy),15,Scalar(0,255,0));
            putText(img,"21",Point(tempx,tempy),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            points[21] = Point(tempx,tempy);
            inshape.close();
            circle(img,Point(0,0),15,Scalar(0,255,0));
            putText(img,"0",Point(0,0),CV_FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,255));
            distance18 = sqrt((points[18].x)*(points[18].x) + (points[18].y)*(points[18].y));
            distance19 = sqrt((points[19].x)*(points[19].x) + (points[19].y)*(points[19].y));
//            cout<<distance18<<endl;
//            cout<<distance19<<endl;
            //========================image transform (begin)====================================
            // 224 224 81 155
//            circle(img,Point(eye_center_x,eye_center_y),15,Scalar(255,0,0));
//            circle(img,Point(up_lip_center_x,up_lip_center_y),15,Scalar(255,0,0));
//            cv::Mat img_transform = sim_transform_image_3channels(new_img, 47, 55,
//                                                                  eye_center_x, eye_center_y,
//                                                                  up_lip_center_x, up_lip_center_y,23,23,20,38);
//            int cal_x;
//            int cal_y;
//            for (int i = 0; i < 21; i++) {
//                cal_x = points[i].x;
//                cal_y = points[i].y;
//                d_points[i].x = ((g_a*cal_x)+(g_b*cal_y)-(g_a*g_tx)-(g_b*g_ty))/((g_a*g_a)+(g_b*g_b));
//                d_points[i].y = (cal_y - (g_b*d_points[i].x) - g_ty)/g_a;
//            }
//            Mat img_crop;;
//            img_transform.copyTo(img_crop);
//            for(int i = 0 ; i < 21;i++) {
//                circle(img_crop, Point(d_points[i].x, d_points[i].y), 1, Scalar(0, 255, 0));
//            }
//            cv::Mat img_transform6 = sim_transform_image_3channels(new_img, new_img.cols, new_img.rows,
//                                                                   eye_center_x, eye_center_y,
//                                                                   up_lip_center_x, up_lip_center_y,23,23,
//                                                                   20,38);
//            resize(img_crop,img_crop,Size(480,400));

//            cv::Mat img_transform = sim_transform_image_3channels(src_img, 224, 224,
//                                                                  eye_center_x, eye_center_y,
//                                                                  up_lip_center_x, up_lip_center_y,112,112,81,155);
            if (distance19>distance18) {
                img_transform2 = sim_transform_image_3channels(src_img, 224, 224,
                                                                       points[18].x, points[18].y,
                                                                       points[16].x, points[16].y,
                                                                       60, 170, 112, 112);
            }
            if (distance18>distance19) {
                img_transform2 = sim_transform_image_3channels(src_img, 224, 224,
                                                                       points[19].x, points[19].y,
                                                                       points[21].x, points[21].y,
                                                                       60, 170, 112, 112);
            }
//            cv::Mat img_transform3 = sim_transform_image_3channels(src_img, 224, 224,
//                                                                   points[21].x, points[21].y,
//                                                                   points[19].x, points[19].y,
//                                                                   60,170,112,112);
//            cv::Mat img_transform4 = sim_transform_image_3channels(src_img, 224, 224,
//                                                                   points[6].x, points[6].y,
//                                                                   points[7].x, points[7].y,
//                                                                   112,112,30,190);
//            cv::Mat img_transform5 = sim_transform_image_3channels(src_img, 224, 224,
//                                                                   points[8].x, points[8].y,
//                                                                   points[9].x, points[9].y,
//                                                                   112,112,85,155);
//            imshow("src_imgage",src_img);
//            imshow("test_image",img);
//            imshow("Face",img_transform);
//            imshow("Left eye",img_transform2);
//            imshow("Right eye",img_transform3);
//            imshow("Nose",img_transform4);
//            imshow("Mouth",img_transform5);
//            waitKey();
//            exit(2);
#ifdef SHOW_TAG
            namedWindow("Display", WINDOW_AUTOSIZE);
			imshow("Display", img_transform);
			waitKey(0);
#endif
//            cout << image_name << endl;
            if (count % 1000 == 0) {
                cout << count << endl;
            }
            boost::filesystem::path path1(parent_dir_save+ "Face/"+image_name);
            path1.remove_filename();
            if (!boost::filesystem::exists(path1))
            if (!boost::filesystem::create_directories(path1)) {
                cerr << "ERROR CREATING DIR" << endl;
                return -1;
            }
            boost::filesystem::path path2(parent_dir_save+ "LeftEye2/"+image_name);
            path2.remove_filename();
            if (!boost::filesystem::exists(path2))
            if (!boost::filesystem::create_directories(path2)) {
                cerr << "ERROR CREATING DIR" << endl;
                return -1;
            }
//            boost::filesystem::path path3(parent_dir_save + "RightEye/" +image_name);
//            path3.remove_filename();
//            if (!boost::filesystem::exists(path3))
//            if (!boost::filesystem::create_directories(path3)) {
//                cerr << "ERROR CREATING DIR" << endl;
//                return -1;
//            }
//            boost::filesystem::path path4(parent_dir_save + "Nose/" +image_name);
//            path4.remove_filename();
//            if (!boost::filesystem::exists(path4))
//            if (!boost::filesystem::create_directories(path4)) {
//                cerr << "ERROR CREATING DIR" << endl;
//                return -1;
//            }
//            boost::filesystem::path path5(parent_dir_save + "Mouth/" +image_name);
//            path5.remove_filename();
//            if (!boost::filesystem::exists(path5))
//            if (!boost::filesystem::create_directories(path5)) {
//                cerr << "ERROR CREATING DIR" << endl;
//                return -1;
//            }
            //cout<<parent_dir_save<<endl;
            //cout<<image_name<<endl;
            b = 0;
            b = image_name.find(".jpg");
            if(b!=-1)
                image_name = image_name.replace(image_name.find(".jpg"), 8, "");
            else
                image_name = image_name.replace(image_name.find(".JPG"), 8, "");
            //cout<<image_name<<endl;
            image_name = image_name.replace(image_name.find("CACD2000/"), 9, "");
            //cout<<image_name<<endl;
            //exit(3);
            //image_name = image_name.replace(image_name.find("morph/Album2/"), 13, "");
//            imwrite(parent_dir_save + "Face/"+ image_name + "_Face.JPG", img_transform);
            imwrite(parent_dir_save + "LeftEye2/" + image_name + "_LeftEye.JPG", img_transform2);
//            imwrite(parent_dir_save + "RightEye/" + image_name + "_RightEye.JPG", img_transform3);
//            imwrite(parent_dir_save + "Nose/" + image_name + "_Nose.JPG", img_transform4);
//            imwrite(parent_dir_save + "Mouth/" + image_name + "_Mouth.JPG", img_transform5);
//            exit(2);
            //========================image transform (end)====================================
        } catch (Exception e) {
            cerr << e.what();
            cout << image_name << endl;
        }
	}
	in.close();

	return 0;
}