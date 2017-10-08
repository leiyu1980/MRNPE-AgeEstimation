#pragma once
#include "stdafx.h"
class _mtimes
{
public:
	LARGE_INTEGER Freq;//64λ�з�������ֵ. // ��ȡʱ������  ��1��/�롱������Hz�����ȣ���1Hz����ÿ��һ��
	LARGE_INTEGER start;  // ��ȡʱ�Ӽ���                
	LARGE_INTEGER end;
	vector<double> step_time;
public:
	_mtimes(){QueryPerformanceFrequency(&Freq);}
	double calcFreq(){return (double)Freq.QuadPart/(double)((end.QuadPart-start.QuadPart));}
	double calcTime()
	{
		/*�˴�*1000���Ժ���Ϊ��λ��*1000000 ��΢��Ϊ��λ*/
		/*����ִ��ʱ�伫�̣������Ǽ�΢�룩�����Բ���΢��Ϊ��λ*/  
		/*  1s=10^3ms(����)=10^6��s(΢��)=10^9ns(����)  */   
		//printf("%d\n",(end.QuadPart-start.QuadPart)*1000000/Freq.QuadPart);
		return (double)((end.QuadPart-start.QuadPart)*1000/Freq.QuadPart);
	}
	double aveFreq()
	{
		double aveFrq=0;
		for (int i=0;i<step_time.size();i++)
			aveFrq+=step_time[i];
		return (double)1000000*(double)step_time.size()/aveFrq;
	}
};