#include "stdafx.h"
#include "model.h"
#include "shapeRegression.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "NPDDetector.h"


int main()
{
	Model model("D:\\Code\\FaceLankmark\\bin\\model_32pts_score.dat");
	string modelFile = "D:\\Code\\FaceLankmark\\bin\\npd_model_fro.dat";

	int numImgs = 1;
	int minObjSize = 80;
	double scaleFactor = 1.2;
	double stepFactor = 0.04;

	Mat img;
	VideoCapture inputVideo(0);
	inputVideo >> img;
	int height = img.rows;
	int width = img.cols;
	cout<<height<<" "<<width<<endl;
	CNPDDetector detector;
	if( !detector.InitDetector(modelFile.data(), height, width, minObjSize, scaleFactor, stepFactor) )
	{
		cout << "Failed to initiate the NPD detector: " << modelFile << endl;
		return 1;
	}
	
	bool b_detect = true;
	namedWindow("detection", CV_WINDOW_AUTOSIZE);
	_bbox bbox;
	shapeXY tShapeXY;

	shapeXY tShapeXY2;

	while(1) {
        inputVideo >> img;

		clock_t begin, finish, procTime = 0;
        
		begin = clock();
		finish = clock();
		if(b_detect)
		{
			detector.BGR2Gray(&img);
			vector<Rect> pResults;
			for(int i = 0; i < numImgs; i++)
			{
				pResults.clear();
				detector.Detect(pResults);
			}
			if(pResults.size()==0)
				continue;

			bbox.rootcoord[0] = pResults[0].x;
			bbox.rootcoord[1] = pResults[0].y;
			bbox.rootcoord[2] = pResults[0].x + pResults[0].width;
			bbox.rootcoord[3] = pResults[0].y + pResults[0].height*1.0;
			b_detect = false;
			DWORD start = GetTickCount();
			float score = 0.0;
			float roll, yaw, pitch;
			shapeRegression(model, img, bbox,tShapeXY2, score, roll, yaw, pitch);
			if(abs(score)>10)
			{
				b_detect = true;
			}
			printf("landmark location time: %dms\n", GetTickCount() - start);
			cout<<score<<endl;
			cout<<"roll: "<<roll<<" yaw: "<<yaw<<" pitch: "<<pitch<<endl;
		}
		else
		{
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
			int center_x = (min_x + max_x)/2;
			int height = max_y - min_y;
			bbox.rootcoord[0] = center_x - height/2;
			bbox.rootcoord[1] = min_y;
			bbox.rootcoord[2] = center_x + height/2;
			bbox.rootcoord[3] = max_y;
			DWORD start = GetTickCount();
			float score = 0.0;
			float roll, yaw, pitch;
			shapeRegression2(model, img, bbox,tShapeXY,tShapeXY2, score, roll, yaw, pitch);
			if(abs(score)>10)
			{
				b_detect = true;
			}
			printf("landmark location time: %dms\n", GetTickCount() - start);
			cout<<score<<endl;
			cout<<"roll: "<<roll<<" yaw: "<<yaw<<" pitch: "<<pitch<<endl;
		}

		
		

		for (int j=0;j<_pointNum;j++)
		{
			circle(img, Point(tShapeXY2.shapeX[j],tShapeXY2.shapeY[j]),1,Scalar(0,255,0), -1, 8);
		}
		tShapeXY = tShapeXY2;

		imshow("detection", img);

        char c = cvWaitKey(50);
		if(c=='r') b_detect = true;
        if(c==27) break;
    }

	return 0;
}