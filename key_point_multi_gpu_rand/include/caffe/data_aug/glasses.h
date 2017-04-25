#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>

using namespace cv;

class CGlasses
{
private:
	vector<Mat> faceVecBGRImg;

	Mat meanMask;
	Mat meanMask_hair;
	Mat oriImg;
	int nLandmarks;

	int ProcessMerge(Mat img,Mat alpha);

	void getGlassesTemplate(Mat oriImg,Mat maskPat,Mat mask,Mat& affinemaskPat,Mat& affinemask,float* keys);

	void getHairTemplate(Mat oriImg,Mat maskPat,Mat mask,Mat& affinemaskPat,Mat& affinemask,float* keys);


public:
	/******************/
	//func:ProcessGlasses()
	//input:faceImg-人脸图像；maskPat-眼镜图案图片；mask-眼镜mask图片；keys-人脸31个关键点
	//output:outImg-戴上眼镜的人脸图片
	/******************/
	void ProcessGlasses(Mat& outImg,Mat faceImg,Mat maskPat,Mat mask,float* keys, int Landmarks);//

	/******************/
	//func:ProcessHair()
	//input:faceImg-人脸图像；maskPat-发型图案图片；mask-发型mask图片；keys-人脸31个关键点
	//output:outImg-戴上发型的人脸图片
	/******************/
	void ProcessHair(Mat& outImg,Mat faceImg,Mat maskPat,Mat mask,float* keys, int Landmarks);//

	void ProcessWaterMark(Mat& outImg,Mat faceImg,Mat mask);

	CGlasses();
	~CGlasses();
};
