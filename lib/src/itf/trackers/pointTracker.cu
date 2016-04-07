#include "itf/trackers/trackers.h"
#include "itf/trackers/gpucommon.hpp"
#include "itf/trackers/utils.h"
#include "thrust/sort.h"
#include <iostream>
#include <stdio.h>
#include <numeric>

//#include <cudnn.h>

using namespace cv;
using namespace cv::gpu;
__device__ int d_framewidth[1],d_frameheight[1];
__device__ int d_buffLen[1], d_tailidx[1],d_total[1];
__device__ int lockOld[NUMTHREAD],lockNew[NUMTHREAD];
__device__ int sobelFilter[3 * 3 * 3];
__device__ unsigned char  x3x3[27], y3x3[27], z3x3[27];
__device__ unsigned char d_clrvec[3*1000];
void setHW(int w, int h)
{
	cudaMemcpyToSymbol(d_framewidth, &w, sizeof(int));
	cudaMemcpyToSymbol(d_frameheight, &h, sizeof(int));
	int tmpsobel[3 * 3 * 3] =
	{
		1, 2, 1, 2, 4, 2, 1, 2, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	   -1,-2,-1,-2,-4,-2,-1,-2,-1
	};
	cudaMemcpyToSymbol(sobelFilter, tmpsobel, sizeof(int) * 27);
	unsigned char tmpz[27] =
	{
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 2, 2
	};
	cudaMemcpyToSymbol(z3x3, tmpz, sizeof(unsigned char) * 27);
	unsigned char tmpy[27] =
	{
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		0, 0, 0, 1, 1, 1, 2, 2, 2
	};
	cudaMemcpyToSymbol(y3x3, tmpy, sizeof(unsigned char) * 27);
	unsigned char tmpx[27] =
	{
		0, 1, 2, 0, 1, 2, 0, 1, 2,
		0, 1, 2, 0, 1, 2, 0, 1, 2,
		0, 1, 2, 0, 1, 2, 0, 1, 2
	};
	cudaMemcpyToSymbol(x3x3, tmpx, sizeof(unsigned char) * 27);
	unsigned char clrve[1000 * 3];
	for(int i=0;i<360;i++)
	{
		HSVtoRGB(clrve + i * 3, clrve + i * 3 + 1, clrve + i * 3 + 2, i+120.0, 1, 1);
	}
	cudaMemcpyToSymbol(d_clrvec, clrve, sizeof(unsigned char) * 3000);
}
texture <unsigned char, cudaTextureType2D, cudaReadModeElementType> volumeTexture(0, cudaFilterModePoint, cudaAddressModeClamp);

#define applyKernel3x3(x,y,z) \
result += sobelFilter[(z[0] * 3 + y[0]) * 3 + x[0]] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z[1] * 3 + y[1]) * 3 + x[1]] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z[2] * 3 + y[2]) * 3 + x[2]] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z[3] * 3 + y[3]) * 3 + x[3]] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z[4] * 3 + y[4]) * 3 + x[4]] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z[5] * 3 + y[5]) * 3 + x[5]] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z[6] * 3 + y[6]) * 3 + x[6]] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z[7] * 3 + y[7]) * 3 + x[7]] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z[8] * 3 + y[8]) * 3 + x[8]] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];\
	\
result += sobelFilter[(z[9] * 3 + y[9]) * 3 + x[9]] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z[10] * 3 + y[10]) * 3 + x[10]] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z[11] * 3 + y[11]) * 3 + x[11]] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z[12] * 3 + y[12]) * 3 + x[12]] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z[13] * 3 + y[13]) * 3 + x[13]] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z[14] * 3 + y[14]) * 3 + x[14]] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z[15] * 3 + y[15]) * 3 + x[15]] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z[16] * 3 + y[16]) * 3 + x[16]] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z[17] * 3 + y[17]) * 3 + x[17]] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];\
	\
result += sobelFilter[(z[18] * 3 + y[18]) * 3 + x[18]] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];\
result += sobelFilter[(z[19] * 3 + y[19]) * 3 + x[19]] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];\
result += sobelFilter[(z[20] * 3 + y[20]) * 3 + x[20]] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];\
result += sobelFilter[(z[21] * 3 + y[21]) * 3 + x[21]] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];\
result += sobelFilter[(z[22] * 3 + y[22]) * 3 + x[22]] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];\
result += sobelFilter[(z[23] * 3 + y[23]) * 3 + x[23]] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];\
result += sobelFilter[(z[24] * 3 + y[24]) * 3 + x[24]] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];\
result += sobelFilter[(z[25] * 3 + y[25]) * 3 + x[25]] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];\
result += sobelFilter[(z[26] * 3 + y[26]) * 3 + x[26]] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];

__global__ void gradientKernel(unsigned char* volume, unsigned char* rgbframe, unsigned char* rframe, unsigned char* gframe, int z0)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	const int z_1 = (z0 - 1 + buffLen) % buffLen;
	const int z1 = (z0 + 1) % buffLen;
	const int xs[3] = { x - 1, x, x + 1 };
	const int ys[3] = { y - 1, y, y + 1 };
	const int zs[3] = { z_1, z0, z1 };

	//for(int i 
	if (x>1 && y>1 && x < fw - 1 && y < fh - 1)
	{
		int offset = y*fw + x;
		int offset3 = offset * 3;
		//rgbframe[offset3] =  abs(rframe[offset]);
		//rgbframe[offset3 + 1] = abs(gframe[offset ]);
		//rgbframe[offset3 + 2] = 0;
		int result = 0, result1=0;
		float val1 = 0;
		/*
		for(int zi =0;zi<3;zi++)
			for (int yi = 0; yi < 3;yi++)
				for (int xi = 0; xi < 3; xi++)
				{
					result += sobelFilter[(zi * 3 + yi) * 3 + xi] * volume[((z-zi+1)*fh+(y+yi-1))*fw + x+xi-1];
					
				}
		*/
		/*
		result += sobelFilter[(0 * 3 + 0) * 3 + 0] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[(0 * 3 + 0) * 3 + 1] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[(0 * 3 + 0) * 3 + 2] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[(0 * 3 + 1) * 3 + 0] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[(0 * 3 + 1) * 3 + 1] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[(0 * 3 + 1) * 3 + 2] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[(0 * 3 + 2) * 3 + 0] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[(0 * 3 + 2) * 3 + 1] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[(0 * 3 + 2) * 3 + 2] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];
		
		result += sobelFilter[(1 * 3 + 0) * 3 + 0] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[(1 * 3 + 0) * 3 + 1] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[(1 * 3 + 0) * 3 + 2] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[(1 * 3 + 1) * 3 + 0] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[(1 * 3 + 1) * 3 + 1] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[(1 * 3 + 1) * 3 + 2] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[(1 * 3 + 2) * 3 + 0] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[(1 * 3 + 2) * 3 + 1] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[(1 * 3 + 2) * 3 + 2] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];

		result += sobelFilter[(2 * 3 + 0) * 3 + 0] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[(2 * 3 + 0) * 3 + 1] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[(2 * 3 + 0) * 3 + 2] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[(2 * 3 + 1) * 3 + 0] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[(2 * 3 + 1) * 3 + 1] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[(2 * 3 + 1) * 3 + 2] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[(2 * 3 + 2) * 3 + 0] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[(2 * 3 + 2) * 3 + 1] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[(2 * 3 + 2) * 3 + 2] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];


		result1 += sobelFilter[(0 * 3 + 0) * 3 + 0] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];
		result1 += sobelFilter[(0 * 3 + 0) * 3 + 1] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];
		result1 += sobelFilter[(0 * 3 + 0) * 3 + 2] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];
		result1 += sobelFilter[(0 * 3 + 1) * 3 + 0] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];
		result1 += sobelFilter[(0 * 3 + 1) * 3 + 1] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];
		result1 += sobelFilter[(0 * 3 + 1) * 3 + 2] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];
		result1 += sobelFilter[(0 * 3 + 2) * 3 + 0] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];
		result1 += sobelFilter[(0 * 3 + 2) * 3 + 1] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];
		result1 += sobelFilter[(0 * 3 + 2) * 3 + 2] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];

		result1 += sobelFilter[(1 * 3 + 0) * 3 + 0] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];
		result1 += sobelFilter[(1 * 3 + 0) * 3 + 1] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];
		result1 += sobelFilter[(1 * 3 + 0) * 3 + 2] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];
		result1 += sobelFilter[(1 * 3 + 1) * 3 + 0] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];
		result1 += sobelFilter[(1 * 3 + 1) * 3 + 1] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];
		result1 += sobelFilter[(1 * 3 + 1) * 3 + 2] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];
		result1 += sobelFilter[(1 * 3 + 2) * 3 + 0] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];
		result1 += sobelFilter[(1 * 3 + 2) * 3 + 1] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];
		result1 += sobelFilter[(1 * 3 + 2) * 3 + 2] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];

		result1 += sobelFilter[(2 * 3 + 0) * 3 + 0] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];
		result1 += sobelFilter[(2 * 3 + 0) * 3 + 1] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];
		result1 += sobelFilter[(2 * 3 + 0) * 3 + 2] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];
		result1 += sobelFilter[(2 * 3 + 1) * 3 + 0] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];
		result1 += sobelFilter[(2 * 3 + 1) * 3 + 1] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		result1 += sobelFilter[(2 * 3 + 1) * 3 + 2] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];
		result1 += sobelFilter[(2 * 3 + 2) * 3 + 0] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];
		result1 += sobelFilter[(2 * 3 + 2) * 3 + 1] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];
		result1 += sobelFilter[(2 * 3 + 2) * 3 + 2] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];
		*/
		float xgrad = 0, ygrad = 0, zgrad = 0;
		applyKernel3x3(x3x3, y3x3, z3x3);
			//rgbframe[offset3] = (abs(result) + abs(result1)) / 16.0;
				zgrad = result / 8.0;
				rgbframe[offset3] = (zgrad);
		result = 0;

		//rgbframe[offset3+1] = volume[offset+fw*fh*z0];
		//rgbframe[offset3+2] = volume[offset + fw*fh*z0];
		
		applyKernel3x3(z3x3, x3x3, y3x3);
			//rgbframe[offset3 + 1] = (abs(result) + abs(result1)) / 16.0;
				ygrad = result / 16.0;
				//rgbframe[offset3 + 1] = abs(ygrad);
		result = 0;
		applyKernel3x3(y3x3, z3x3, x3x3);
			//rgbframe[offset3 + 2] = (abs(result) + abs(result1)) / 16.0;
				xgrad = result / 16.0;
				//rgbframe[offset3 + 2] = abs(xgrad);
		result = 0;
		
		float grad = sqrt(xgrad*xgrad + ygrad*ygrad + zgrad*zgrad);
		//if (x<fw / 2)result = abs(result);

		
		rgbframe[offset3] = grad;
		
		rgbframe[offset3 + 1] = volume[offset + fw*fh*z0];
		rgbframe[offset3 + 2] = volume[offset + fw*fh*z0];
		
	}
}

__global__ void FastSobel_t(unsigned char* volume,unsigned char* volumeRGB, int z0, float* gradient
	,unsigned char* rgb_showFrame)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int z_1 = (z0 - 1 + buffLen) % buffLen;
	const int z1 = (z0 + 1) % buffLen;
	const int xs[3] = { x - 1, x, x + 1 };
	const int ys[3] = { y - 1, y, y + 1 };
	const int zs[3] = { z_1, z0, z1 };
	unsigned char* rgbframe = volumeRGB;// + z0*fh*fw*3;
	//for(int i 
	if (x>1 && y>1 && x < fw - 1 && y < fh - 1)
	{
		int offset = y*fw + x;
		int offset3 = offset * 3;
		//rgbframe[offset3] =  abs(rframe[offset]);
		//rgbframe[offset3 + 1] = abs(gframe[offset ]);
		//rgbframe[offset3 + 2] = 0;
		int result = 0, result1 = 0;
		float val1 = 0;
		
		result += sobelFilter[0] * volume[(zs[0] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[1] * volume[(zs[0] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[2] * volume[(zs[0] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[3] * volume[(zs[0] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[4] * volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[5] * volume[(zs[0] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[6] * volume[(zs[0] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[7] * volume[(zs[0] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[8] * volume[(zs[0] * fh + ys[2])*fw + xs[2]];
		/*
		result += sobelFilter[9] * volume[(zs[1] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[10] * volume[(zs[1] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[11] * volume[(zs[1] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[12] * volume[(zs[1] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[13] * volume[(zs[1] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[14] * volume[(zs[1] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[15] * volume[(zs[1] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[16] * volume[(zs[1] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[17] * volume[(zs[1] * fh + ys[2])*fw + xs[2]];
		*/
		result += sobelFilter[18] * volume[(zs[2] * fh + ys[0])*fw + xs[0]];
		result += sobelFilter[19] * volume[(zs[2] * fh + ys[0])*fw + xs[1]];
		result += sobelFilter[20] * volume[(zs[2] * fh + ys[0])*fw + xs[2]];
		result += sobelFilter[21] * volume[(zs[2] * fh + ys[1])*fw + xs[0]];
		result += sobelFilter[22] * volume[(zs[2] * fh + ys[1])*fw + xs[1]];
		result += sobelFilter[23] * volume[(zs[2] * fh + ys[1])*fw + xs[2]];
		result += sobelFilter[24] * volume[(zs[2] * fh + ys[2])*fw + xs[0]];
		result += sobelFilter[25] * volume[(zs[2] * fh + ys[2])*fw + xs[1]];
		result += sobelFilter[26] * volume[(zs[2] * fh + ys[2])*fw + xs[2]];
		float grad = abs(result / 8.0);
		int index = min(int(grad),300);
		unsigned char r = d_clrvec[index * 3], g = d_clrvec[index * 3+1] , b = d_clrvec[index * 3+2];;
		gradient[offset] = grad;
		//val1 = volume[(zs[2] * fh + ys[1])*fw + xs[1]] - volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		float alpha = min(1.0,double(grad / 300));
		//rgb_showFrame[offset3] = rgbframe[offset3] * (1-alpha) + r*alpha;
		//rgb_showFrame[offset3 + 1] = rgbframe[offset3 + 1] * (1 - alpha) + g*alpha;
		//rgb_showFrame[offset3 + 2] = rgbframe[offset3 + 2] * (1 - alpha) + b*alpha;
	}
}
__global__ void FastSobel_RGB( unsigned char* volume, int z0, float* gradient, unsigned char* rgb_showFrame)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int z_1 = (z0 - 1 + buffLen) % buffLen;
	const int z1 = (z0 + 1) % buffLen;
	const int xs[3] = { x - 1, x, x + 1 };
	const int ys[3] = { y - 1, y, y + 1 };
	const int zs[3] = { z_1, z0, z1 };
	unsigned char* rgbframe = volume + z0*fh*fw*3;

	if (x>1 && y>1 && x < fw - 1 && y < fh - 1)
	{
		int offset = y*fw + x;
		int offset3 = offset * 3;
		//rgbframe[offset3] =  abs(rframe[offset]);
		//rgbframe[offset3 + 1] = abs(gframe[offset ]);
		//rgbframe[offset3 + 2] = 0;
		int result = 0, result1 = 0,result2=0;
		float val1 = 0;

		result += sobelFilter[0] * volume[((zs[0] * fh + ys[0])*fw + xs[0]) * 3];
		result += sobelFilter[1] * volume[((zs[0] * fh + ys[0])*fw + xs[1]) * 3];
		result += sobelFilter[2] * volume[((zs[0] * fh + ys[0])*fw + xs[2]) * 3];
		result += sobelFilter[3] * volume[((zs[0] * fh + ys[1])*fw + xs[0]) * 3];
		result += sobelFilter[4] * volume[((zs[0] * fh + ys[1])*fw + xs[1]) * 3];
		result += sobelFilter[5] * volume[((zs[0] * fh + ys[1])*fw + xs[2]) * 3];
		result += sobelFilter[6] * volume[((zs[0] * fh + ys[2])*fw + xs[0]) * 3];
		result += sobelFilter[7] * volume[((zs[0] * fh + ys[2])*fw + xs[1]) * 3];
		result += sobelFilter[8] * volume[((zs[0] * fh + ys[2])*fw + xs[2]) * 3];

		result += sobelFilter[18] * volume[((zs[2] * fh + ys[0])*fw + xs[0]) * 3];
		result += sobelFilter[19] * volume[((zs[2] * fh + ys[0])*fw + xs[1]) * 3];
		result += sobelFilter[20] * volume[((zs[2] * fh + ys[0])*fw + xs[2]) * 3];
		result += sobelFilter[21] * volume[((zs[2] * fh + ys[1])*fw + xs[0]) * 3];
		result += sobelFilter[22] * volume[((zs[2] * fh + ys[1])*fw + xs[1]) * 3];
		result += sobelFilter[23] * volume[((zs[2] * fh + ys[1])*fw + xs[2]) * 3];
		result += sobelFilter[24] * volume[((zs[2] * fh + ys[2])*fw + xs[0]) * 3];
		result += sobelFilter[25] * volume[((zs[2] * fh + ys[2])*fw + xs[1]) * 3];
		result += sobelFilter[26] * volume[((zs[2] * fh + ys[2])*fw + xs[2]) * 3];

		result1 += sobelFilter[0] * volume[((zs[0] * fh + ys[0])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[1] * volume[((zs[0] * fh + ys[0])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[2] * volume[((zs[0] * fh + ys[0])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[3] * volume[((zs[0] * fh + ys[1])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[4] * volume[((zs[0] * fh + ys[1])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[5] * volume[((zs[0] * fh + ys[1])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[6] * volume[((zs[0] * fh + ys[2])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[7] * volume[((zs[0] * fh + ys[2])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[8] * volume[((zs[0] * fh + ys[2])*fw + xs[2]) * 3 + 1];

		result1 += sobelFilter[18] * volume[((zs[2] * fh + ys[0])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[19] * volume[((zs[2] * fh + ys[0])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[20] * volume[((zs[2] * fh + ys[0])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[21] * volume[((zs[2] * fh + ys[1])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[22] * volume[((zs[2] * fh + ys[1])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[23] * volume[((zs[2] * fh + ys[1])*fw + xs[2]) * 3 + 1];
		result1 += sobelFilter[24] * volume[((zs[2] * fh + ys[2])*fw + xs[0]) * 3 + 1];
		result1 += sobelFilter[25] * volume[((zs[2] * fh + ys[2])*fw + xs[1]) * 3 + 1];
		result1 += sobelFilter[26] * volume[((zs[2] * fh + ys[2])*fw + xs[2]) * 3 + 1];

		result2 += sobelFilter[0] * volume[((zs[0] * fh + ys[0])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[1] * volume[((zs[0] * fh + ys[0])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[2] * volume[((zs[0] * fh + ys[0])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[3] * volume[((zs[0] * fh + ys[1])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[4] * volume[((zs[0] * fh + ys[1])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[5] * volume[((zs[0] * fh + ys[1])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[6] * volume[((zs[0] * fh + ys[2])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[7] * volume[((zs[0] * fh + ys[2])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[8] * volume[((zs[0] * fh + ys[2])*fw + xs[2]) * 3 + 2];

		result2 += sobelFilter[18] * volume[((zs[2] * fh + ys[0])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[19] * volume[((zs[2] * fh + ys[0])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[20] * volume[((zs[2] * fh + ys[0])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[21] * volume[((zs[2] * fh + ys[1])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[22] * volume[((zs[2] * fh + ys[1])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[23] * volume[((zs[2] * fh + ys[1])*fw + xs[2]) * 3 + 2];
		result2 += sobelFilter[24] * volume[((zs[2] * fh + ys[2])*fw + xs[0]) * 3 + 2];
		result2 += sobelFilter[25] * volume[((zs[2] * fh + ys[2])*fw + xs[1]) * 3 + 2];
		result2 += sobelFilter[26] * volume[((zs[2] * fh + ys[2])*fw + xs[2]) * 3 + 2];


		float grad0 = abs(result / 8.0), grad1 = abs(result1 / 8.0), grad2 = abs(result2 / 8.0);
		//int index = min(int(grad), 300);
		//unsigned char r = d_clrvec[index * 3], g = d_clrvec[index * 3 + 1], b = d_clrvec[index * 3 + 2];;
		float grad = (grad0 + grad1 + grad2) / 3;
		gradient[offset] = grad;
		//val1 = volume[(zs[2] * fh + ys[1])*fw + xs[1]] - volume[(zs[0] * fh + ys[1])*fw + xs[1]];
		float alpha =  min(1.0, double(grad / 300));
		if (offset > 0 && offset < fh*fw)
		{
			rgb_showFrame[offset3] = grad0*0.5;
			rgb_showFrame[offset3 + 1] =  grad1*0.5;
			rgb_showFrame[offset3 + 2] =  grad2*0.5;
		}
	}
}
void CrowdTracker::calcGradient()
{
	debuggingFile << "calcGradient:" << std::endl;
	cudaMemcpyToSymbol(d_buffLen, &buffLen, sizeof(int));
	cudaMemcpyToSymbol(d_tailidx,&tailidx, sizeof(int));
	/*
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
	cudaSafeCall(cudaBindTexture2D(0, volumeTexture, volume->gpuAt(frameSize*4), &desc, img.cols, img.rows, img.step));
	*/
	dim3 block(32, 32);
	dim3 grid(divUp(frame_width, 32), divUp(frame_height, 32));
	//FastSobel_t << < grid, block >> >(volumeGray->gpu_ptr(), volumeRGB->gpuAt(mididx*frameSizeRGB) , mididx, gradient->gpu_ptr(), d_rgbframedata);
	FastSobel_RGB << < grid, block >> >(volumeRGB->gpu_ptr(), mididx, gradient->gpu_ptr(), d_rgbframedata);
	gpu::GpuMat gradMat(frame_height, frame_width, CV_32FC1, gradient->gpu_ptr());

}
__global__ void getGradPoints(float* gradient, int2* cornerBuff, unsigned char* mask,unsigned char* d_rgbframedata)
{
	int buffLen = d_buffLen[0];
	int fw = d_framewidth[0];
	int fh = d_frameheight[0];
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < fw&&y < fh)
	{
		
		int offset = y*fw + x;
		if (mask[offset])
		{
			float val = gradient[offset];
			if (val > 100)
			{
				int posidx = atomicAdd(d_total, 1);
				cornerBuff[posidx].x = x;
				cornerBuff[posidx].y = y;
				//d_rgbframedata[offset * 3] = 0;
				//d_rgbframedata[offset * 3 + 1] = 128;
				//d_rgbframedata[offset * 3 + 2] = 255;
			}
		}
	}
}
__global__ void pick_GradPoints(unsigned char* pointRange, int2* cornerBuff,int2* corners)
{
	int addidx = blockIdx.x;
	int ptidx = threadIdx.x;
	int h = d_frameheight[0], w = d_framewidth[0];
	__shared__ unsigned char good[1];
	good[0] = 0;
	__syncthreads();
	if (d_total[0] < blockDim.x)
	{
		int x0 = cornerBuff[addidx].x,y0=cornerBuff[addidx].y;
		if (ptidx < d_total[0])
		{
			int xi = corners[ptidx].x, yi = corners[ptidx].y;
			int dist = abs(x0 - xi) + abs(y0 - yi);
			int range = pointRange[yi*w+xi];
			if (dist < range)
			{  
				good[0]=1;
			}
		}
		__syncthreads();
		if (threadIdx.x==0&&!good[0])
		{
			int posidx = atomicAdd(d_total, 1);
			if (posidx < blockDim.x)
			{
				corners[posidx].x = x0;
				corners[posidx].y = y0;
			}
		}
	}

}

void CrowdTracker::findPoints()
{
	dim3 block(32, 32);
	dim3 grid(divUp(frame_width, 32), divUp(frame_height, 32));
	totalGradPoint = 0;
	
	cudaMemcpyToSymbol(d_total, &totalGradPoint, sizeof(int));
	getGradPoints <<< grid, block >>>(gradient->gpu_ptr(), cornerBuff->gpu_ptr(), mask->gpu_ptr(), d_rgbframedata);
	
	cudaMemcpyFromSymbol(&totalGradPoint, d_total, sizeof(int));
	debuggingFile << "totalGradPoint:"<<totalGradPoint << std::endl;
	cornerBuff->SyncD2H();
	
	acceptPtrNum = 0;
	
	unsigned char* rangeMat = detector->rangeMat.data;
	int2* tmpptr = cornerBuff->cpu_ptr();
	int addpos = 0;
	int2* addptr = corners->cpu_ptr();
	float fp2[2];
	for (int i = 0; i<totalGradPoint; i++)
	{
		int x = tmpptr->x, y = tmpptr->y;
		tmpptr ++;
		uchar range = rangeMat[y*frame_width+x];
		int rangeval = range*range;
		bool good = true;
		int2* ptr2 = corners->cpu_ptr();
		for (int j = 0; j<addpos; j++)
		{
			int x1 = ptr2->x, y1 = ptr2->y;
			int dx = x1 - x, dy = y1 - y;
			if (dx*dx<rangeval&&dy*dx<rangeval)    
			{
				goto findNewPoint;
			}
			ptr2++;
		}
		memcpy(addptr, tmpptr,sizeof(int2));
		addptr++;
		addpos++;
		if (addpos >= nSearch)
			break;
findNewPoint:
	}
	acceptPtrNum = addpos;
	corners->SyncH2D();
	debuggingFile << "acceptPtrNum:" << acceptPtrNum << std::endl;
	/*
	if (totalGradPoint>0 && totalGradPoint<65535)
	{
		cudaMemcpyToSymbol(d_total, &acceptPtrNum, sizeof(int));
		debuggingFile << "totalGradPoint : " << totalGradPoint << std::endl;
		pick_GradPoints << < totalGradPoint, nSearch >> >(pointRange->gpu_ptr(), cornerBuff->gpu_ptr(), corners->gpu_ptr());
		cudaMemcpyFromSymbol(&acceptPtrNum, d_total, sizeof(int));
		corners->SyncD2H();
	}

	debuggingFile << "total Find: " << acceptPtrNum << std::endl;
	*/
	
}

__global__ void filterTracks(TracksInfo trkinfo, uchar* status, float2* update_ptr, float* d_persMap)
{
	int idx = threadIdx.x;
	int len = trkinfo.lenVec[idx];
	bool flag = status[idx];
	float x = update_ptr[idx].x, y = update_ptr[idx].y;
	int frame_width = d_framewidth[0], frame_heigh = d_frameheight[0];
	trkinfo.nextTrkptr[idx].x = x;
	trkinfo.nextTrkptr[idx].y = y;
	float curx = trkinfo.curTrkptr[idx].x, cury = trkinfo.curTrkptr[idx].y;
	float dx = x - curx, dy = y - cury;
	float dist = sqrt(dx*dx + dy*dy);
	float cumDist = dist + trkinfo.curDistPtr[idx];
	trkinfo.nextDistPtr[idx] = cumDist;
	if (flag&&len>0)
	{

		int xb = x + 0.5, yb = y + 0.5;
		UperLowerBound(xb, 0, frame_width-1);
		UperLowerBound(yb, 0, frame_heigh - 1);
		float persval = d_persMap[yb*frame_width + xb];
		if (xb < 10 && yb < 10)flag = false;
		//        int prex=trkinfo.curTrkptr[idx].x+0.5, prey=trkinfo.curTrkptr[idx].y+0.5;
		//        int trkdist=abs(prex-xb)+abs(prey-yb);
		float trkdist = abs(dx) + abs(dy);
		if (trkdist>persval)
		{
			flag = false;
		}
		//printf("%d,%.2f,%d|",trkdist,persval,flag);
		int Movelen = 150 / sqrt(persval);
		Movelen = 15;
		//Movelen is the main factor wrt perspective
		//        printf("%d\n",Movelen);
		if (flag&&Movelen<len)
		{
			//            int offset = (tailidx+bufflen-Movelen)%bufflen;
			//            FeatPts* dataptr = next_ptr-tailidx*NQue;
			//            FeatPts* aptr = dataptr+offset*NQue;
			//            float xa=aptr[idx].x,ya=aptr[idx].y;
			FeatPts* ptr = trkinfo.getPtr_(trkinfo.trkDataPtr, idx, Movelen);
			float xa = ptr->x, ya = ptr->y;
			float displc = sqrt((x - xa)*(x - xa) + (y - ya)*(y - ya));
			float curveDist = cumDist - *(trkinfo.getPtr_(trkinfo.distDataPtr, idx, Movelen));
			//if(persval*0.1>displc)
			if ( displc<3)
			{
				flag = false;
			}
		}
	}
	int newlen = flag*(len + (len<trkinfo.buffLen));
	trkinfo.lenVec[idx] = newlen;
	if (newlen>minTrkLen)
	{
		FeatPts* pre_ptr = trkinfo.preTrkptr;
		float prex = pre_ptr[idx].x, prey = pre_ptr[idx].y;
		float vx = (x - prex) / minTrkLen, vy = (y - prey) / minTrkLen;
		float spd = sqrt(vx*vx + vy*vy);
		trkinfo.nextSpdPtr[idx] = spd;
		trkinfo.nextVeloPtr[idx].x = vx, trkinfo.nextVeloPtr[idx].y = vy;
	}
}

void CrowdTracker::filterTrackGPU()
{
	debuggingFile << "filterTrackGPU" << std::endl;
	trkInfo = tracksGPU->getInfoGPU();
	trkInfo.preTrkptr = trkInfo.getVec_(trkInfo.trkDataPtr, minTrkLen - 1);
	
	filterTracks <<< 1, nFeatures >>>(trkInfo, gpuStatus.data, (float2 *)gpuNextPts.data, persMap->gpu_ptr());
	tracksGPU->increPtr();
	trkInfo = tracksGPU->getInfoGPU();
	trkInfo.preTrkptr = trkInfo.getVec_(trkInfo.trkDataPtr, minTrkLen);
	debuggingFile << "Finshed filterTrackGPU" << std::endl;
}

__global__ void  addNewPts(FeatPts* cur_ptr, int* lenVec, int2* new_ptr, float2* nextPtrs)
{
	int idx = threadIdx.x;
	int dim = blockDim.x;
	__shared__ int counter[1];
	counter[0] = 0;
	__syncthreads();
	//printf("(%d)", idx);
	if (lenVec[idx] <= 0)
	{
		int posidx = atomicAdd(counter, 1);
		
		if (posidx<dim)
		{
			float x = new_ptr[posidx].x, y = new_ptr[posidx].y;
			cur_ptr[idx].x = x;
			cur_ptr[idx].y = y;
			lenVec[idx] += 1;
			
		}
	}

	nextPtrs[idx].x = cur_ptr[idx].x;
	nextPtrs[idx].y = cur_ptr[idx].y;
	//__syncthreads();
	//d_total[0] = counter[0];
}

__global__ void applyPointPersMask(unsigned char* d_mask, FeatPts* cur_ptr, int* lenVec, float* d_persMap)
{
	int pidx = blockIdx.x;
	int len = lenVec[pidx];
	if (len>0)
	{
		float px = cur_ptr[pidx].x, py = cur_ptr[pidx].y;
		int blocksize = blockDim.x;
		int w = d_framewidth[0], h = d_frameheight[0];
		int localx = threadIdx.x, localy = threadIdx.y;
		int pxint = px + 0.5, pyint = py + 0.5;
		UperLowerBound(pyint, 0, h - 1);
		UperLowerBound(pxint, 0, w - 1);
		float persval = d_persMap[pyint*w + pxint];
		float range = Pers2Range(persval)+1;
		int offset = range + 0.5;
		int yoffset = localy - blocksize / 2;
		int xoffset = localx - blocksize / 2;
		
		if (abs(yoffset)<range&&abs(xoffset)<range)
		{
			int globalx = xoffset + pxint, globaly = yoffset + pyint;
			int globaloffset = globaly*w + globalx;
			if (globaloffset < w*h && globaloffset>0)
			{
				//printf("%d)", globaloffset);
				d_mask[globaloffset] = 0;
			}
		}
	}
}
void CrowdTracker::PersExcludeMask()
{
	addNewPts << <1, nFeatures, 0, cornerStream >> >(tracksGPU->curTrkptr, tracksGPU->lenVec, corners->gpu_ptr(), (float2*)gpuPrePts.data);
	cudaMemcpyAsync(mask->gpu_ptr(), roimask->gpu_ptr(), frame_height*frame_width*sizeof(unsigned char), cudaMemcpyDeviceToDevice, cornerStream);
	dim3 block(32, 32, 1);
	applyPointPersMask << <nFeatures, block, 0, cornerStream >> >(mask->gpu_ptr(), tracksGPU->curTrkptr, tracksGPU->lenVec, persMap->gpu_ptr());
	//corners->SyncD2HStream(cornerStream);
	/*
	addNewPts << <1, nFeatures >> >(tracksGPU->curTrkptr, tracksGPU->lenVec, corners->gpu_ptr(), (float2*)gpuPrePts.data);
	debuggingFile << "there" << std::endl;
	std::cout << std::endl;
	cudaMemcpy(mask->gpu_ptr(), roimask->gpu_ptr(), frame_height*frame_width*sizeof(unsigned char), cudaMemcpyDeviceToDevice);
	dim3 block(32, 32, 1);
	applyPointPersMask << <nFeatures, block>> >(mask->gpu_ptr(), tracksGPU->curTrkptr, tracksGPU->lenVec, persMap->gpu_ptr());
	*/
}
__global__ void renderFrame( unsigned char* d_frameptr, int totallen, unsigned char* d_mask)
{
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int offset3 = offset * 3;
	if (offset<totallen)
	{
		if (d_mask[offset])
		{
			d_frameptr[offset3] *=  0.5;
			d_frameptr[offset3 + 1] *=  0.5;
			d_frameptr[offset3 + 2] *=  0.5;
		}
	}
}
void CrowdTracker::Render(unsigned char* framedata)
{
	int nblocks = frameSize / 1024;
	//renderMask->toZeroD();
	debuggingFile << "render" << std::endl;
	renderFrame << <nblocks, 1024 >> >(d_rgbframedata, frameSize, mask->gpu_ptr());
	//cudaMemcpy(framedata, rgbMat.data, frame_height*frame_width * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

__global__ void rgb2grayKernel(unsigned char * d_frameRGB, unsigned char* d_frameGray,int total)
{
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int offset3 = offset*3;
	if (offset < total)
	{
		float r = d_frameRGB[offset3], g = d_frameRGB[offset3+1],b=d_frameRGB[offset3+2];
		d_frameGray[offset] = 0.299*r + 0.587*g + 0.114*b;
	}
}
void CrowdTracker::RGB2Gray(unsigned char * d_frameRGB,unsigned char* d_frameGray)
{
	int nblocks = frameSize / 1024+1;
	rgb2grayKernel << <nblocks, 1024 >> >(d_frameRGB, d_frameGray, frameSize);
}
__global__ void clearLostStats(int* lenVec, int* d_neighbor, float* d_cosine, float* d_velo, float* d_distmat, int nFeatures)
{
	int c = threadIdx.x, r = blockIdx.x;
	if (r<nFeatures, c<nFeatures)
	{
		bool flag1 = (lenVec[c]>0), flag2 = (lenVec[r]>0);
		bool flag = flag1&&flag2;
		if (!flag)
		{

			d_neighbor[r*nFeatures + c] = 0;
			d_neighbor[c*nFeatures + r] = 0;
			d_cosine[r*nFeatures + c] = 0;
			d_cosine[c*nFeatures + r] = 0;
			d_velo[r*nFeatures + c] = 0;
			d_velo[c*nFeatures + r] = 0;
			d_distmat[r*nFeatures + c] = 0;
			d_distmat[c*nFeatures + r] = 0;
		}
	}
}
__global__ void searchNeighbor(TracksInfo trkinfo,
	int* d_neighbor, float* d_cosine, float* d_velo, float* d_distmat,
	float * d_persMap, int nFeatures)
{
	int c = threadIdx.x, r = blockIdx.x;
	int clen = trkinfo.lenVec[c], rlen = trkinfo.lenVec[r];
	FeatPts* cur_ptr = trkinfo.curTrkptr;
	if (clen>minTrkLen&&rlen>minTrkLen&&r<c)
	{
		//        int offset = (tailidx+bufflen-minTrkLen)%bufflen;
		//        FeatPts* pre_ptr=data_ptr+NQue*offset;
		//        FeatPts* pre_ptr=trkinfo.preTrkptr;//trkinfo.getVec_(trkinfo.trkDataPtr,minTrkLen-1);
		//        float cx0=pre_ptr[c].x,cy0=pre_ptr[c].y;
		//        float rx0=pre_ptr[r].x,ry0=pre_ptr[r].y;
		float cx1 = cur_ptr[c].x, cy1 = cur_ptr[c].y;
		float rx1 = cur_ptr[r].x, ry1 = cur_ptr[r].y;
		float dx = abs(rx1 - cx1), dy = abs(ry1 - cy1);
		float dist = sqrt(dx*dx + dy*dy);
		int  ymid = (ry1 + cy1) / 2.0 + 0.5, xmid = (rx1 + cx1) / 2.0 + 0.5;
		float persval = 0;
		int ymin = min(ry1, cy1), xmin = min(rx1, cx1);
		persval = d_persMap[ymin*d_framewidth[0] + xmin];
		float hrange = persval, wrange = persval;
		if (hrange<2)hrange = 2;
		if (wrange<2)wrange = 2;
		float distdecay = 0.05, cosdecay = 0.1, velodecay = 0.05;
		/*
		float vx0 = rx1 - rx0, vx1 = cx1 - cx0, vy0 = ry1 - ry0, vy1 = cy1 - cy0;
		float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
		float veloCo = abs(norm0-norm1)/(norm0+norm1);
		float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;
		*/
		float vrx = trkinfo.curVeloPtr[r].x, vry = trkinfo.curVeloPtr[r].y
			, vcx = trkinfo.curVeloPtr[c].x, vcy = trkinfo.curVeloPtr[c].y;
		float normr = trkinfo.curSpdPtr[r], normc = trkinfo.curSpdPtr[c];
		float veloCo = abs(normr - normc) / (normr + normc);
		float cosine = (vrx*vcx + vry*vcy) / normr / normc;
		dist = wrange*1.5 / (dist + 0.01);
		dist = 2 * dist / (1 + abs(dist)) - 1;
		//dist=-((dist > wrange) - (dist < wrange));
		d_distmat[r*nFeatures + c] = dist + d_distmat[r*nFeatures + c] * (1 - distdecay);
		d_distmat[c*nFeatures + r] = dist + d_distmat[c*nFeatures + r] * (1 - distdecay);
		d_cosine[r*nFeatures + c] = cosine + d_cosine[r*nFeatures + c] * (1 - cosdecay);
		d_cosine[c*nFeatures + r] = cosine + d_cosine[c*nFeatures + r] * (1 - cosdecay);
		d_velo[r*nFeatures + c] = veloCo + d_velo[r*nFeatures + c] * (1 - velodecay);
		d_velo[c*nFeatures + r] = veloCo + d_velo[c*nFeatures + r] * (1 - velodecay);
		if (d_distmat[r*nFeatures + c]>5 && d_cosine[r*nFeatures + c]>1)//&&d_velo[r*nFeatures+c]<(14*velodecay)*0.9)
		{
			d_neighbor[r*nFeatures + c] += 1;
			d_neighbor[c*nFeatures + r] += 1;
		}
		else
		{
			d_neighbor[r*nFeatures + c] /= 2.0;
			d_neighbor[c*nFeatures + r] /= 2.0;
		}

	}
}
void CrowdTracker::pointCorelate()
{
	clearLostStats << <nFeatures, nFeatures >> >(tracksGPU->lenData->gpu_ptr(),nbCount->gpu_ptr(), cosCo->gpu_ptr(), veloCo->gpu_ptr(), distCo->gpu_ptr(), nFeatures);
	searchNeighbor << <nFeatures, nFeatures >> >(trkInfo, nbCount->gpu_ptr(), cosCo->gpu_ptr(), veloCo->gpu_ptr(), distCo->gpu_ptr(), persMap->gpu_ptr(), nFeatures);

}
