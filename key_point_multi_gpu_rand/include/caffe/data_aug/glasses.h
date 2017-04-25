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
	//input:faceImg-����ͼ��maskPat-�۾�ͼ��ͼƬ��mask-�۾�maskͼƬ��keys-����31���ؼ���
	//output:outImg-�����۾�������ͼƬ
	/******************/
	void ProcessGlasses(Mat& outImg,Mat faceImg,Mat maskPat,Mat mask,float* keys, int Landmarks);//

	/******************/
	//func:ProcessHair()
	//input:faceImg-����ͼ��maskPat-����ͼ��ͼƬ��mask-����maskͼƬ��keys-����31���ؼ���
	//output:outImg-���Ϸ��͵�����ͼƬ
	/******************/
	void ProcessHair(Mat& outImg,Mat faceImg,Mat maskPat,Mat mask,float* keys, int Landmarks);//

	void ProcessWaterMark(Mat& outImg,Mat faceImg,Mat mask);

	CGlasses();
	~CGlasses();
};
