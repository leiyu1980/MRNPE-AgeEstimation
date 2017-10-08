// Class for the NPD detector
// 
// Shengcai Liao
// National Laboratory of Pattern Recognition
// Institute of Automation, Chinese Academy of Sciences
// scliao@nlpr.ia.ac.cn
//

#include "stdafx.h"
#include "NPDDetector.h"
#include <omp.h>
#include <math.h>
#include <iostream>

using namespace std;


CNPDDetector::CNPDDetector()
{
	mppImg = 0; // �Ҷ�ͼ��洢�ռ�
	mHeight = 0; // �����ͼ��߶�
	mWidth = 0; // �����ͼ����	
	mpBlocks = 0; // ����
	mNumBlocks = 0; // ������Ŀ
}


CNPDDetector::~CNPDDetector()
{
	if ( mppImg )
	{
		if ( mppImg[0] ) delete[] mppImg[0];
		delete[] mppImg;
	}

	if(mpBlocks) delete[] mpBlocks;
}



// ��ʼ��NPD�����
// filename��ѵ���õļ��ģ���ļ���maxObjSize�����Ŀ���С��minObjSize����СĿ���С��
// scaleFactor����߶ȼ��ĳ߶����ӣ�stepFactor�������������ӣ���������������ڼ�ⴰ��С�ı�����
bool CNPDDetector::InitDetector(const char* filename, int height, int width, int minObjSize, double scaleFactor, double stepFactor)
{
	mHeight = height;
	mWidth = width;

	// ��ȡѵ���õļ��ģ��
	if (!mNpdModel.ReadModel(filename, min(height,width), minObjSize, scaleFactor, stepFactor)) return false;

	// ����Ҷ�ͼ�洢�ռ�
	mppImg = new unsigned char*[height];
	if(!mppImg)
	{
		// cout << "Out of memory: mppImg" << endl;
		return false;
	}

	mppImg[0] = new unsigned char[height * width];
	if(!mppImg[0])
	{
		// cout << "Out of memory: mppImg" << endl;
		return false;
	}

	for(int i = 1; i < height; i++) mppImg[i] = mppImg[i-1] + width;

	if(!InitBlocks()) return false;

	// ����NPD���ұ�
	InitLookupTable();

	return true;
}


void CNPDDetector::InitLookupTable() // ����NPD���ұ�
{	
	mpppGrayTable = vector< vector< vector<unsigned char> > >(256);

	for(int i = 0; i < 256; i++)
	{
		mpppGrayTable[i] = vector< vector<unsigned char> >(256);
		for(int j = 0; j < 256; j++) mpppGrayTable[i][j] = vector<unsigned char>(256);
	}

	double cr = 0.299, cg = 0.587, cb = 0.114;

	for(int r = 0; r < 256; r++)
	{
		for(int g = 0; g < 256; g++)
		{
			for(int b = 0; b < 256; b++)
			{
				mpppGrayTable[r][g][b] = (unsigned char)floor(cr * r + cg * g + cb * b + 0.5);
			}
		}
	}

	for (int x = 0; x < 256; x++)
	{
		for (int y = 0; y < 256; y++)
		{
			if (x == 0 && y == 0) mppNPDLookupTable[x][y] = 128;
			else
			{
				float val =  float(x) / float(x + y);
				// ��Ҫ�Ӹ���������Ϊ8λ����
				val = floor(256 * val);
				if(val > 255) val = 255;
				mppNPDLookupTable[x][y] = (unsigned char) val;
			}
		}
	}
}
	

bool CNPDDetector::InitBlocks() // ��ʼ���������
{
	mNumBlocks = 0;

	// ���������Ŀ
	for (int s = 0; s < mNpdModel.mNumScales; s++)
	{
		int winStep = mNpdModel.mpWinStep[s];
		int winSize = mNpdModel.mpWinSize[s];
		int rowMax = mHeight - winSize + 1;
		int colMax = mWidth - winSize + 1;

		mNumBlocks += ( (rowMax-1) / winStep + 1) * ( (colMax-1) / winStep + 1); // ����������Զ�ȡ�̶�������
	}

	// cout << "Total scanning subwindows: " << mNumBlocks << endl << endl;

	// �����ڴ�
	mpBlocks = new CBlock[mNumBlocks];
	if(mpBlocks == 0)
	{
		// cout << "Out of memory: mpBlocks" << endl;
		return false;
	}
	
	int k = 0;

	// ����������
	for (int s = 0; s < mNpdModel.mNumScales; s++)
	{
		int winStep = mNpdModel.mpWinStep[s];
		int winSize = mNpdModel.mpWinSize[s];
		int rowMax = mHeight - winSize + 1;
		int colMax = mWidth - winSize + 1;

		// process each subwindow
		for (int r = 0; r < rowMax; r += winStep) // slide in row
		{
			for (int c = 0; c < colMax; c += winStep) // slide in column
			{
				mpBlocks[k].scale = s;
				mpBlocks[k].rect.x = c;
				mpBlocks[k].rect.y = r;
				mpBlocks[k].rect.height = winSize;
				mpBlocks[k].rect.width = winSize;
				mpBlocks[k].pData = mppImg[r] + c;

				k++;
			}
		}
	}

	// ����������Ӧ���ص�һά����
	for (int i = 0; i < mNpdModel.mNumBranchNodes; i++)
	{
		for (int s = 0; s < mNpdModel.mNumScales; s++)
		{
			mNpdModel.mpBranchNode[i].pFea[s].index1 = mNpdModel.mpBranchNode[i].pFea[s].y1 * mWidth + mNpdModel.mpBranchNode[i].pFea[s].x1;
			mNpdModel.mpBranchNode[i].pFea[s].index2 = mNpdModel.mpBranchNode[i].pFea[s].y2 * mWidth + mNpdModel.mpBranchNode[i].pFea[s].x2;
		}
	}

	return true;
}
	

// ��ɫͼ��ת�Ҷȣ�ע������ͼ����BGR˳��ģ�
void CNPDDetector::BGR2Gray(Mat *frame)
{
#pragma omp parallel for

	for(int i = 0; i < frame->rows; i++)
	{
		unsigned char* pImg = frame->ptr<unsigned char>(i);

		for(int j = 0; j < frame->cols; j++)
		{
			//mppImg[i][j] = unsigned char( floor(cb * pImg[0] + cg * pImg[1] + cr * pImg[2] + 0.5) );
			mppImg[i][j] = mpppGrayTable[pImg[2]][pImg[1]][pImg[0]];
			pImg += 3;
		}
	}
}


// ��⺯���������ȵ���BGR2Gray��
// ppImg������ͼ�񣬶�ά���飬��ppImg[i][j]��ʾ��i�е�j�����أ�height, width��ͼ��ߴ磻pResults: �������
bool CNPDDetector::Detect(vector<Rect>& pResults)
{
// #pragma omp parallel for

	// �������м���
	for (int k = 0; k < mNumBlocks; k++)
	{
		unsigned char *pData = mpBlocks[k].pData;
		int scale = mpBlocks[k].scale;
		int *pStageThreshold = mNpdModel.mpStageThreshold;
		CBranchNode **ppTreeRoot = mNpdModel.mppTreeRoot;
		//double score = 0;
		int fx = 0;
		int t;

		for (t = 0; t < mNpdModel.mNumStages; t++)
		{
			CBranchNode *pBranchNode = *ppTreeRoot++;

			// test the current tree classifier
			while (true)
			{
				CNPDFeature *pFea = pBranchNode->pFea + scale;

				unsigned char feaVal = mppNPDLookupTable[ pData[pFea->index1] ][ pData[pFea->index2] ];

				if (feaVal < pBranchNode->cutpoint[0] || feaVal > pBranchNode->cutpoint[1]) // ������
				{
					if (pBranchNode->isLeftABranch == true) // branch node
					{
						pBranchNode = (CBranchNode *) pBranchNode->pLeftChild;
					}
					else // leaf node
					{
						fx = fx + *(int *) pBranchNode->pLeftChild;
						break;
					}
				}
				else // ������
				{
					if (pBranchNode->isRightABranch == true) // branch node
					{
						pBranchNode = (CBranchNode *) pBranchNode->pRightChild;
					}
					else // leaf node
					{
						fx = fx + *(int *) pBranchNode->pRightChild;
						break;
					}
				}
			}

			if (fx < *pStageThreshold++) break; // negative samples
		}

		if (t == mNpdModel.mNumStages) // a face detected
		{
			//pBlock->score = score;

// #pragma omp critical // modify the record by a single thread
			{
				pResults.push_back( mpBlocks[k].rect );
			}
		}
	}

	groupRectangles(pResults, 3);

	return true;
}
