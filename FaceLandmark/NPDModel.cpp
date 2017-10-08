// Class for the NPD detection model
// 
// Shengcai Liao
// National Laboratory of Pattern Recognition
// Institute of Automation, Chinese Academy of Sciences
// scliao@nlpr.ia.ac.cn
//

#include "stdafx.h"
#include "NPDModel.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;


CNPDModel::CNPDModel(void)
{
	mObjSize = 0; // ģ���С
	mNumStages = 0; // ����������DQT��Ŀ
	mNumBranchNodes = 0; // ����������֧�ڵ��ܸ���
	mNumLeafNodes = 0; // ��������Ҷ�ӽڵ��ܸ���
	mpStageThreshold = 0; // ÿ��ľ�����ֵ����Ҫ�Ӹ���������QUAN_LEVELS����Ϊ��������mNumStages��1
	mppTreeRoot = 0; // // ÿ�����ĸ��ڵ�ָ�루ָ��mpBranchNode�е�ĳЩ�ڵ㣬��ָ��Ľڵ㼴Ϊ���ĸ��ڵ㣩��mNumTrees��1
	mpBranchNode = 0; // ��֧�ڵ��б�mNumBranchNodes��1
	mpFit = 0; // ÿ��Ҷ�ӽڵ��Ԥ��ֵ����Ҫ�Ӹ���������QUAN_LEVELS����Ϊ��������mNumLeafNodes��1

	mNumScales = 0; // ��߶ȼ��ĳ߶ȸ���
	mpScaleFactor = 0; // ��߶ȼ��ĳ߶����ӣ�ģ��Ŵ�ϵ������mNumScales��1
	mpWinSize = 0; // ��߶ȼ��ļ�ⴰ�ڴ�С��mNumScales��1
	mpWinStep = 0; // ��ⴰ����������mNumScales��1
}


CNPDModel::~CNPDModel(void)
{
	if(mpStageThreshold != 0) delete[] mpStageThreshold;
	if(mppTreeRoot != 0) delete[] mppTreeRoot;
	if(mpFit != 0) delete[] mpFit;
	if(mpScaleFactor != 0) delete[] mpScaleFactor;
	if (mpWinSize != 0) delete[] mpWinSize;
	if (mpWinStep != 0) delete[] mpWinStep;

	if(mpBranchNode != 0)
	{
		for(int i = 0; i < mNumBranchNodes; i++) // ɾ����߶�NPD���������洢�ռ�
		{
			delete[] mpBranchNode[i].pFea;
		}

		delete[] mpBranchNode;
	}
}


// ��ȡѵ���õļ��ģ��
// filename��ѵ���õļ��ģ���ļ���maxObjSize�����Ŀ���С��minObjSize����СĿ���С��
// scaleFactor����߶ȼ��ĳ߶����ӣ�stepFactor�������������ӣ���������������ڼ�ⴰ��С�ı�����
bool CNPDModel::ReadModel(const char* filename, int maxObjSize, int minObjSize, double scaleFactor, double stepFactor)
{
	if(scaleFactor <= 1.0f || minObjSize <= 0 || maxObjSize < minObjSize)
	{
		// cout << "Error input parameters: scaleFactor <= 1.0f || minObjSize <= 0 || maxObjSize < minObjSize." << endl;
		return false;
	}

	ifstream fin(filename, ios::binary);
	if(fin.fail())
	{
		// cout << "Cannot open model file: " << filename << endl;
		return false;
	}

	// cout << "Load NPD detection model from " << filename << endl;

	fin.read((char*) &mObjSize, sizeof(int)); // ģ���С
	fin.read((char*) &mNumStages, sizeof(int)); // ����������DQT��Ŀ
	if(fin.fail())
	{
		// cout << "Error reading the model file for mObjSize and mNumStages: " << filename << endl;
		fin.close();
		return false;
	}

	if(mObjSize <= 0 || mNumStages <= 0)
	{
		// cout << "Error model parameters: mObjSize <= 0 || mNumStages <= 0." << endl;
		fin.close();
		return false;
	}

	fin.read((char*) &mNumBranchNodes, sizeof(int)); // ����������֧�ڵ��ܸ���
	fin.read((char*) &mNumLeafNodes, sizeof(int)); // ��������Ҷ�ӽڵ��ܸ���
	if(fin.fail())
	{
		// cout << "Error reading the model file for mNumBranchNodes, and mNumLeafNodes: " << filename << endl;
		fin.close();
		return false;
	}

	if(mNumBranchNodes <= 0 || mNumLeafNodes <= 0)
	{
		// cout << "Error model parameters: mNumBranchNodes <= 0 || mNumLeafNodes <= 0." << endl;
		fin.close();
		return false;
	}

	mpStageThreshold = new int [mNumStages]; // ÿ��ľ�����ֵ����Ҫ�Ӹ���������QUAN_LEVELS����Ϊ��������mNumStages��1
	if(mpStageThreshold == 0)
	{
		// cout << "Out of memory: mpStageThreshold." << endl;
		fin.close();
		return false;
	}
	

	for ( int i = 0; i < mNumStages; i++)
	{
		float threshold;
		fin.read((char*) &threshold, sizeof(float));
		if(fin.fail())
		{
			// cout << "Error reading the model file for mpStageThreshold: " << filename << endl;
			fin.close();
			return false;
		}
		
		// �任��ֵ����߱ȶ��ٶ�
		//threshold = float ( log( threshold / (1.0 - (double)threshold) ) / 2.0 );
		// ÿ��ľ�����ֵ����Ҫ�Ӹ���������QUAN_LEVELS����Ϊ������
		mpStageThreshold[i] = (int) floor(double(threshold) * QUAN_LEVELS);
	}

	mppTreeRoot = new CBranchNode* [mNumStages]; // ÿ�����ĸ��ڵ�ָ�루ָ��mpBranchNode�е�ĳЩ�ڵ㣬��ָ��Ľڵ㼴Ϊ���ĸ��ڵ㣩��mNumStages��1
	mpBranchNode = new CBranchNode[mNumBranchNodes]; // ��֧�ڵ��б�mNumBranchNodes��1
	mpFit = new int[mNumLeafNodes]; // ÿ��Ҷ�ӽڵ��Ԥ��ֵ��mNumLeafNodes��1
	if (mppTreeRoot == 0 || mpBranchNode == 0 || mpFit == 0)
	{
		// cout << "Out of memory: mppTreeRoot, mpBranchNode, mpFit." << endl;
		fin.close();
		return false;
	}

	for (int i = 0; i < mNumStages; i++)
	{
		int rootIndex;
		fin.read((char*)&rootIndex, sizeof(int));
		if (fin.fail())
		{
			// cout << "Error reading the model file for tree root index: " << filename << endl;
			fin.close();
			return false;
		}

		mppTreeRoot[i] = mpBranchNode + rootIndex;
	}


	minObjSize = max(mObjSize, minObjSize); //��С��ⴰ�ڴ�С�������ģ�崰�ڴ�С
	mNumScales = (int) floor((log(maxObjSize) - log(minObjSize)) / log(scaleFactor)) + 1; // ȷ����߶ȼ��ĳ߶ȸ���

	for(int i = 0; i < mNumBranchNodes; i++) // ���ٶ�߶�NPD���������洢�ռ�
	{
		mpBranchNode[i].pFea = new CNPDFeature[mNumScales];
		if(mpBranchNode[i].pFea == 0)
		{
			// cout << "Out of memory: mpBranchNode[i].pFea." << endl;
			fin.close();
			return false;
		}
	}

	for(int i = 0; i < mNumBranchNodes; i++) // ��pixel1
	{
		short int pixel;
		fin.read((char*)&pixel, sizeof(short int));
		if(fin.fail())
		{
			// cout << "Error reading the model file for pixel1: " << filename << endl;
			fin.close();
			return false;
		}

		mpBranchNode[i].pFea[0].x1 = pixel / mObjSize; //�Զ�floorȡ��
		mpBranchNode[i].pFea[0].y1 = pixel % mObjSize;
	}

	for (int i = 0; i < mNumBranchNodes; i++) // ��pixel2
	{
		short int pixel;
		fin.read((char*)&pixel, sizeof(short int));
		if (fin.fail())
		{
			// cout << "Error reading the model file for pixel2: " << filename << endl;
			fin.close();
			return false;
		}
		
		mpBranchNode[i].pFea[0].x2 = pixel / mObjSize; //�Զ�floorȡ��
		mpBranchNode[i].pFea[0].y2 = pixel % mObjSize;
	}

	for (int i = 0; i < mNumBranchNodes; i++) // ���ֲ�ڵ�ķֲ���ֵ
	{
		fin.read((char*)&mpBranchNode[i].cutpoint, 2 * sizeof(unsigned char));
		if (fin.fail())
		{
			// cout << "Error reading the model file for mpBranchNode[i].cutpoint: " << filename << endl;
			fin.close();
			return false;
		}
	}

	int child;

	for (int i = 0; i < mNumBranchNodes; i++) // ����������ָ��
	{
		fin.read((char*)&child, sizeof(int));
		if (fin.fail())
		{
			// cout << "Error reading the model file for left child: " << filename << endl;
			fin.close();
			return false;
		}

		if (child >= 0) // ָ��ֲ�ڵ�
		{
			mpBranchNode[i].isLeftABranch = true;
			mpBranchNode[i].pLeftChild = mpBranchNode + child;
		}
		else // ָ��Ҷ�ӽڵ�
		{
			mpBranchNode[i].isLeftABranch = false;
			mpBranchNode[i].pLeftChild = mpFit - child - 1;
		}
	}

	for (int i = 0; i < mNumBranchNodes; i++) // ����������ָ��
	{
		fin.read((char*)&child, sizeof(int));
		if (fin.fail())
		{
			// cout << "Error reading the model file for right child: " << filename << endl;
			fin.close();
			return false;
		}

		if (child >= 0) // ָ��ֲ�ڵ�
		{
			mpBranchNode[i].isRightABranch = true;
			mpBranchNode[i].pRightChild = mpBranchNode + child;
		}
		else // ָ��Ҷ�ӽڵ�
		{
			mpBranchNode[i].isRightABranch = false;
			mpBranchNode[i].pRightChild = mpFit - child - 1;
		}
	}

	for( int i = 0; i < mNumLeafNodes; i++ )
	{
		float fit = 0;
		fin.read((char*)&fit, sizeof(float)); // ��Ҷ�ӽڵ��Ԥ��ֵ
		if (fin.fail())
		{
			// cout << "Error reading the model file for mpFit: " << filename << endl;
			fin.close();
			return false;
		}
		
		// ÿ��Ҷ�ӽڵ��Ԥ��ֵ����Ҫ�Ӹ���������QUAN_LEVELS����Ϊ������
		mpFit[i] = (int) floor(double(fit) * QUAN_LEVELS);;
	}

	fin.close();
	
//	cout << "Succefully loaded the NPD detection model." << endl << endl
//		<< "Model details: " << endl
//		<< "objSize: " << mObjSize << endl
//		<< "#stages: " << mNumStages << endl
//		<< "#branches: " << mNumBranchNodes << endl
//		<< "#leaves: " << mNumLeafNodes << endl << endl
//		<< "Detection parameters: " << endl
//		<< "maxObjSize = " << maxObjSize << endl
//		<< "minObjSize = " << minObjSize << endl
//		<< "scaleFactor = " << scaleFactor << endl
//		<< "stepFactor = " << stepFactor << endl << endl;


	mpScaleFactor = new double[mNumScales]; // ��߶ȼ��ĳ߶����ӣ�ģ��Ŵ�ϵ������mNumScales��1
	mpWinSize = new int[mNumScales]; // ��߶ȼ��ļ�ⴰ�ڴ�С��mNumScales��1
	mpWinStep = new int[mNumScales]; // ��ⴰ����������mNumScales��1
	if (mpScaleFactor == 0 || mpWinSize == 0 || mpWinStep == 0)
	{
		// cout << "Out of memory: mpScaleFactor, mpWinSize, mpWinStep." << endl;
		fin.close();
		return false;
	}

	for (int s = 0; s < mNumScales; s++)
	{
		mpWinSize[s] = (int) floor(minObjSize * pow(scaleFactor, s) + 0.5); // ���õ�����scaleFactor��Ϊ�˱���ȡ�����Ŵ�
		mpScaleFactor[s] = double(mpWinSize[s]) / double(mObjSize);

		// ȷ����ⴰ��������
		mpWinStep[s] = (int)floor(mpWinSize[s] * stepFactor);
		if (mpWinStep[s] < 1) mpWinStep[s] = 1;
	}

	// ��չ�����Ϊ��߶ȼ����
	for (int i = 0; i < mNumBranchNodes; i++)
	{
		for (int s = 1; s < mNumScales; s++)
		{
			mpBranchNode[i].pFea[s].x1 = (short) floor(mpBranchNode[i].pFea[0].x1 * mpScaleFactor[s] + 0.5);
			mpBranchNode[i].pFea[s].y1 = (short) floor(mpBranchNode[i].pFea[0].y1 * mpScaleFactor[s] + 0.5);
			mpBranchNode[i].pFea[s].x2 = (short) floor(mpBranchNode[i].pFea[0].x2 * mpScaleFactor[s] + 0.5);
			mpBranchNode[i].pFea[s].y2 = (short) floor(mpBranchNode[i].pFea[0].y2 * mpScaleFactor[s] + 0.5);
		}
	}

	return true;
}
