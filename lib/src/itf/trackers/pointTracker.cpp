#include "itf/trackers/trackers.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>

#include <cuda_runtime.h>
#include "itf/trackers/gpucommon.hpp"
#include "itf/trackers/utils.h"
using namespace cv;
using namespace cv::gpu;


CrowdTracker::CrowdTracker()
{
    frame_width=0, frame_height=0;
    frameidx=0;
    nFeatures=0,nSearch=0; 
    /**cuda **/
    persDone=false;
}
CrowdTracker::~CrowdTracker()
{
    releaseMemory();
}
void setHW(int w,int h);
int CrowdTracker::init(int w, int h,unsigned char* framedata,int nPoints)
{
    /** Checking Device Properties **/
    int nDevices;
    int maxthread=0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
		std::cout << "maxThreadsPerBlock:" << prop.maxThreadsPerBlock << std::endl;
        

        //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,MyKernel, 0, arrayCount);
        if(maxthread==0)maxthread=prop.maxThreadsPerBlock;
        //debuggingFile << prop.major << "," << prop.minor << std::endl;
    }
	cudaSetDevice(1);
	std::cout <<"device Status:"<< cudaSetDevice(1) << std::endl;
    /** Basic **/
    frame_width = w,frame_height = h;
	frameSize = frame_width*frame_height;
	frameSizeRGB = frame_width*frame_height*3;
	tailidx = 0, buffLen = 10, mididx = 0, preidx = 0, nextidx = 0;
    setHW(w,h);
    frameidx=0;

	volumeRGB = new MemBuff<unsigned char>(frameSizeRGB*buffLen);
	volumeGray = new MemBuff<unsigned char>(frameSize*buffLen);
	gpu_zalloc(d_rgbframedata, frameSizeRGB, sizeof(unsigned char));
	rgbMat = gpu::GpuMat(frame_height, frame_width, CV_8UC3, d_rgbframedata);
	gpuPreRGBA = gpu::GpuMat(frame_height, frame_width, CV_8UC4);
	gpuRGBA = gpu::GpuMat(frame_height, frame_width, CV_8UC4);

    persMap =  new MemBuff<float>(frame_height*frame_width);
    gpuPersMap= gpu::GpuMat(frame_height, frame_width, CV_32F ,persMap->gpu_ptr());
    roimask =  new MemBuff<unsigned char>(frame_height*frame_width);
    roiMaskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,roimask->gpu_ptr());
	debuggingFile.open("trackerDump.txt", std::ofstream::out);

    /** Point Tracking and Detecting **/
	nFeatures = maxthread;//(maxthread>1024)?1024:maxthread;
	nFeatures = (maxthread>nPoints) ? nPoints : maxthread;
	nSearch = nFeatures;
	tracksGPU = new Tracks();
	tracksGPU->init(nFeatures, nFeatures);
	//detector=new  gpu::GoodFeaturesToTrackDetector_GPU(nSearch,1e-30,0,3);
	detector = new TargetFinder(nSearch, 1e-30, 3);
	tracker = new  gpu::PyrLKOpticalFlow();
	tracker->winSize = Size(9, 9);
	tracker->maxLevel = 3;
	tracker->iters = 10;

	corners = new MemBuff<int2>(nSearch);
	cornerBuff = new MemBuff<int2>(frameSize);
	totalGradPoint = 0, acceptPtrNum = 0;
	cornerCVStream = gpu::Stream();
	cornerStream = gpu::StreamAccessor::getStream(cornerCVStream);
	detector->_stream = cornerCVStream;
	detector->_stream_t = cornerStream;
	cudaStreamCreate(&CorrStream);

	
	gpuCorners = gpu::GpuMat(1, nSearch, CV_32SC2, corners->gpu_ptr());

	mask = new MemBuff<unsigned char>(frame_height*frame_width);
	maskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1, mask->gpu_ptr());
	pointRange = new MemBuff<unsigned char>(frame_height*frame_width);
	status = new MemBuff<unsigned char>(nFeatures);
    gpuStatus=gpu::GpuMat(1,nFeatures,CV_8UC1,status->gpu_ptr());
	statusMat = Mat(1, nFeatures, CV_8UC1, status->cpu_ptr());

	
	prePts = Mat(1, nFeatures, CV_32FC2);
	nextPts = Mat(1, nFeatures, CV_32FC2);
	gpuNextPts = GpuMat(1, nFeatures, CV_32FC2);
	gpuPrePts = GpuMat(1, nFeatures, CV_32FC2);


	gradient = new MemBuff<float>(frameSize, CV_32FC1);

	//Neighbor Search
	nbCount = new MemBuff<int>(nFeatures*nFeatures);
	distCo = new MemBuff<float>(nFeatures*nFeatures);
	cosCo = new MemBuff<float>(nFeatures*nFeatures);
	veloCo = new MemBuff<float>(nFeatures*nFeatures);
	correlation = new MemBuff<float>(nFeatures);
    /** Self Init **/
    selfinit(framedata);

    debuggingFile<< "inited" << std::endl;
    return 1;
}
int CrowdTracker::selfinit(unsigned char* framedata)
{
	Mat curframe(frame_height, frame_width, CV_8UC3, framedata);
	rgbMat.upload(curframe);
	gpuPreGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(mididx*frameSize));
	gpuGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(nextidx*frameSize));
	gpu::cvtColor(rgbMat, gpuGray, CV_RGB2GRAY);


	unsigned char* tmpPtr = volumeGray->gpu_ptr() + tailidx*frameSize;
	cudaMemcpy(tmpPtr, gpuGray.data, frameSize, cudaMemcpyDeviceToDevice);
	tmpPtr = volumeGray->gpu_ptr() + tailidx*frameSizeRGB;
	cudaMemcpy(tmpPtr, rgbMat.data, frameSizeRGB, cudaMemcpyDeviceToDevice);
	tailidx = (tailidx + 1) % buffLen;
	mididx = (tailidx + buffLen / 2) % buffLen;
	preidx = (mididx - 1 + buffLen) % buffLen;
	nextidx = (mididx + 1 + buffLen) % buffLen;
	return true;
}

int CrowdTracker::updateAframe(unsigned char* framedata, int fidx)
{
    std::clock_t start=std::clock();
    curStatus=FINE;
    frameidx=fidx;
    debuggingFile<<"frameidx:"<<frameidx<<std::endl;
    //Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
	cudaMemcpy(d_rgbframedata,framedata, frameSizeRGB, cudaMemcpyHostToDevice);
	unsigned char* tmpPtr = volumeGray->gpu_ptr() + tailidx*frameSize;
	//GpuMat tmpMat(frame_height, frame_width, CV_8UC1, tmpPtr);
	RGB2Gray(d_rgbframedata, tmpPtr);
	//gpu::cvtColor(rgbMat, tmpMat, CV_RGB2GRAY);
	tmpPtr = volumeRGB->gpu_ptr() + tailidx*frameSizeRGB;
	cudaMemcpy(tmpPtr, d_rgbframedata, frameSizeRGB, cudaMemcpyDeviceToDevice);
	mididx = (tailidx + buffLen / 2) % buffLen;
	preidx = (mididx - 1 + buffLen) % buffLen;
	nextidx = (mididx + 1 + buffLen) % buffLen;
	tailidx = (tailidx + 1) % buffLen;
	cudaMemcpy(d_rgbframedata, volumeRGB->gpuAt(mididx*frameSizeRGB), frameSizeRGB, cudaMemcpyDeviceToDevice);
	calcGradient();

	gpuPreGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(preidx*frameSize));
	gpuGray = GpuMat(frame_height, frame_width, CV_8UC1, volumeGray->gpuAt(mididx*frameSize));
	/* rgba KLT tracking
	gpuPreRGB = GpuMat(frame_height, frame_width, CV_8UC3, volumeRGB->gpuAt(preidx*frameSizeRGB));
	gpuRGB = GpuMat(frame_height, frame_width, CV_8UC3, volumeRGB->gpuAt(mididx*frameSizeRGB));
	debuggingFile << "cvtColor"<< std::endl;
	gpu::cvtColor(gpuPreRGB, gpuPreRGBA,CV_RGB2RGBA);
	gpu::cvtColor(gpuRGB, gpuRGBA, CV_RGB2RGBA);
	debuggingFile << "finish cvtColor" << std::endl;
	*/ 
	
	findPoints();
	PointTracking();
	filterTrackGPU();
	
	
	/*
	pointCorelate();
	nbCount->SyncD2H();
	*/

	PersExcludeMask();
	Render(framedata);
	cudaMemcpy(framedata,d_rgbframedata,frameSizeRGB,cudaMemcpyDeviceToHost);
	tracksGPU->Sync();
	debuggingFile << "frame End" << std::endl;
    return 1;
}

void CrowdTracker::PointTracking()
{
	debuggingFile << "tracker" << std::endl;
	tracker->sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus);
	//tracker->sparse(gpuPreRGBA, gpuRGBA, gpuPrePts, gpuNextPts, gpuStatus);
}
void CrowdTracker::releaseMemory()
{
	tracker->releaseMemory();
	gpuGray.release();
	gpuPreGray.release();
	rgbMat.release();
	gpuPrePts.release();
	gpuNextPts.release();
	gpuStatus.release();
}


void CrowdTracker::setUpPersMap(float* srcMap)
{
	/*
	// camera calibration
	for(int y=0;y<frame_height;y++)
	for(int x=0;x<frame_width;x++)
	{
	float cdist=(frame_width/2.0-abs(x-frame_width/2.0))/frame_width*10;
	srcMap[y*frame_width+x]=srcMap[y*frame_width+x]+cdist*cdist;
	}
	*/
	persMap->updateCPU(srcMap);
	persMap->SyncH2D();
	detector->setPersMat(gpuPersMap, frame_width, frame_height);
	cudaMemcpy(pointRange->gpu_ptr(),detector->rangeMat.data,frameSize,cudaMemcpyHostToDevice);
}
void CrowdTracker::updateROICPU(float* aryPtr, int length)
{
	roimask->toZeroD();
	roimask->toZeroH();
	unsigned char* h_roimask = roimask->cpu_ptr();
	std::vector<Point2f> roivec;
	int counter = 0;
	for (int i = 0; i<length; i++)
	{
		Point2f p(aryPtr[i * 2], aryPtr[i * 2 + 1]);
		roivec.push_back(p);
	}
	for (int i = 0; i<frame_height; i++)
	{
		for (int j = 0; j<frame_width; j++)
		{
			if (pointPolygonTest(roivec, Point2f(j, i), true)>0)
			{
				h_roimask[i*frame_width + j] = 255;
				counter++;

			}
		}
	}

	debuggingFile << counter << std::endl;
	roimask->SyncH2D();
}
void CrowdTracker::updateROImask(unsigned char * ptr)
{
	roimask->toZeroD();
	roimask->toZeroH();
	unsigned char* h_roimask = roimask->cpu_ptr();
	memcpy(h_roimask, ptr, roimask->byte_size);
	roimask->SyncH2D();
}