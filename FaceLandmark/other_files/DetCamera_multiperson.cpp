#include "stdafx.h"
#include "model.h"
#include "shapeRegression.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "NPDDetector.h"

#define TRACKING 1

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

bool merge(Rect r1, Rect r2)
{
	 float x1 = r1.x;
	 float y1 = r1.y;
	 float width1 = r1.width;
	 float height1 = r1.height;

	 float x2 = r2.x;
	 float y2 = r2.y;
	 float width2 = r2.width;
	 float height2 = r2.height;

	 float endx = max(x1+width1, x2+width2);
	 float startx = min(x1,x2);
	 float width = width1 + width2 - (endx-startx);

	 float endy = max(y1+height1, y2+height2);
	 float starty = min(y1,y2);
	 float height = height1 + height2 - (endy-starty);

	 bool D; 
	 
	 float Area, Area1, Area2, ratio;
	 if (width<=0 || height<=0)
		 D = false;
	 else
	 {
		 Area = width*height;
		 Area1 = width1*height1;
		 Area2 = width2*height2;
		 ratio = Area/(Area1+Area2-Area);
		 if (ratio>=0.5)
			 D = true;
		 else
			 D = false;
	 }

	 return D;
}


int main3()
{
	Model model("D:\\Code\\FaceLankmark\\bin\\model_32pts_score_fast.dat");
	string modelFile = "D:\\Code\\FaceLankmark\\bin\\npd_model_fro.dat";

	int numImgs = 10;
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
	
	namedWindow("detection", CV_WINDOW_AUTOSIZE);
	_bbox bbox;
	shapeXY tShapeXY;

	shapeXY tShapeXY2;

	vector<Rect> pResults1;
	pResults1.clear();

	vector<shapeXY> tShapeXY_buffer1;
	tShapeXY_buffer1.clear();


	while(1) {
        inputVideo >> img;

		clock_t begin, finish, procTime = 0;
        
		begin = clock();
		finish = clock();

		DWORD start = GetTickCount();
		detector.BGR2Gray(&img);
		vector<shapeXY> tShapeXY_buffer;
		tShapeXY_buffer.clear();
		vector<Rect> pResults0;
		pResults0.clear();
		detector.Detect(pResults0);

		#if TRACKING
		vector<Rect> pResults;
		pResults.clear();

		//add tracking results
		for(int i=0; i<pResults1.size(); i++)
		{
			bbox.rootcoord[0] = pResults1[i].x;
			bbox.rootcoord[1] = pResults1[i].y;
			bbox.rootcoord[2] = pResults1[i].x + pResults1[i].width;
			bbox.rootcoord[3] = pResults1[i].y + pResults1[i].height*1.0;

			DWORD start = GetTickCount();
			float score = 0.0;
			float roll, yaw, pitch;
			shapeRegression2(model, img, bbox,tShapeXY_buffer1[i],tShapeXY2, score, roll, yaw, pitch);
			//cout<<score<<endl;
			int min_x = 9999.0;
			int min_y = 9999.0;
			int max_x = -9999.0;
			int max_y = -9999.0;
			for(int j=0; j<_pointNum; j++)
			{
				if(tShapeXY2.shapeX[j]<min_x)
					min_x = tShapeXY2.shapeX[j];
				if(tShapeXY2.shapeX[j]>max_x)
					max_x = tShapeXY2.shapeX[j];
				if(tShapeXY2.shapeY[j]<min_y)
					min_y = tShapeXY2.shapeY[j];
				if(tShapeXY2.shapeY[j]>max_y)
					max_y = tShapeXY2.shapeY[j];
			}
			int center_x = (min_x + max_x)/2;
			int height = max_y - min_y;

			Rect pResults_temp;
			pResults_temp.height = height;
			pResults_temp.width = max_y - min_y;
			pResults_temp.x = center_x - height/2;
			pResults_temp.y = min_y;

			if(abs(score)<10)
			{
				pResults.push_back(pResults_temp);
				tShapeXY_buffer.push_back(tShapeXY2);
			}
		}

		//detection
		for(int i=0; i<pResults0.size(); i++)
		{
			bbox.rootcoord[0] = pResults0[i].x;
			bbox.rootcoord[1] = pResults0[i].y;
			bbox.rootcoord[2] = pResults0[i].x + pResults0[i].width;
			bbox.rootcoord[3] = pResults0[i].y + pResults0[i].height*1.0;

			DWORD start = GetTickCount();
			float score = 0.0;
			float roll, yaw, pitch;
			shapeRegression(model, img, bbox,tShapeXY2, score, roll, yaw, pitch);
			//cout<<score<<endl;

			int min_x = 9999.0;
			int min_y = 9999.0;
			int max_x = -9999.0;
			int max_y = -9999.0;
			for(int j=0; j<_pointNum; j++)
			{
				if(tShapeXY2.shapeX[j]<min_x)
					min_x = tShapeXY2.shapeX[j];
				if(tShapeXY2.shapeX[j]>max_x)
					max_x = tShapeXY2.shapeX[j];
				if(tShapeXY2.shapeY[j]<min_y)
					min_y = tShapeXY2.shapeY[j];
				if(tShapeXY2.shapeY[j]>max_y)
					max_y = tShapeXY2.shapeY[j];
			}
			int center_x = (min_x + max_x)/2;
			int height = max_y - min_y;

			Rect pResults_temp;
			pResults_temp.height = height;
			pResults_temp.width = max_y - min_y;
			pResults_temp.x = center_x - height/2;
			pResults_temp.y = min_y;

			if(abs(score)<10)
			{
				pResults.push_back(pResults_temp);
				tShapeXY_buffer.push_back(tShapeXY2);
			}
		}

		

		pResults1.clear();
		tShapeXY_buffer1.clear();

		//ºÏ²¢¼ì²â¿ò
		for(int i=0; i<pResults.size(); i++)
		{
			bool do_merge = false;
			for(int j=0; j<pResults1.size(); j++)
			{
				if(merge(pResults[i], pResults1[j]))
				{
					do_merge = true;
					break;
				}
			}

			if (!do_merge)
			{
				pResults1.push_back(pResults[i]);
				tShapeXY_buffer1.push_back(tShapeXY_buffer[i]);
			}

		}

		#endif
	 
	 #if !TRACKING
		pResults1 = pResults0;
     #endif

		printf("face detection time: %dms\n", GetTickCount() - start);

		for (int i=0;i<pResults1.size();i++)
		{
			rectangle(img, Point(pResults1[i].x, pResults1[i].y), 
				Point(pResults1[i].x + pResults1[i].width, pResults1[i].y + pResults1[i].height), CV_RGB(0,255,0), 2);
		}
		for (int i=0;i<pResults1.size();i++)
		{
			for (int j=0;j<_pointNum;j++)
			{
				circle(img, Point(tShapeXY_buffer1[i].shapeX[j],tShapeXY_buffer1[i].shapeY[j]),1,Scalar(0,255,0), -1, 8);
			}
		}

		imshow("detection", img);

        char c = cvWaitKey(50);
        if(c==27) break;
    }

	return 0;
}