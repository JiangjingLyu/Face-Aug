#include "caffe/data_aug/glasses.h"

CGlasses::CGlasses()
{
	meanMask = Mat::zeros(2,2,CV_64FC1);
	meanMask.at<double>(0,0) = 177;
	meanMask.at<double>(0,1) = 208;
	meanMask.at<double>(1,0) = 269;
	meanMask.at<double>(1,1) = 208;

	meanMask_hair = Mat::zeros(2,2,CV_64FC1);
	meanMask_hair.at<double>(0,0) = 160;
	meanMask_hair.at<double>(0,1) = 208;
	meanMask_hair.at<double>(1,0) = 288;
	meanMask_hair.at<double>(1,1) = 208;
}
CGlasses::~CGlasses()
{

}

void CGlasses::getGlassesTemplate(Mat oriImg,Mat maskPat,Mat mask,Mat& affinemaskPat,Mat& affinemask,float* keys)
{
	
	affinemask= Mat::zeros( oriImg.rows, oriImg.cols, oriImg.type());
	affinemaskPat = Mat::zeros( oriImg.rows, oriImg.cols, oriImg.type());

	vector<Point2d> affinePts;
	Mat m = meanMask;
	//vector<Point2d> vecMeanMask = Mat_<Point2d>(m);  
	vector<Point2f> srcTri(2);
	vector<Point2f> dstTri(2);
	//srcTri[0] = Point2f(vecMeanMask[0 ]); srcTri[1] = Point2f(vecMeanMask[1]);
	srcTri[0] = Point2f(meanMask.at<double>(0,0),meanMask.at<double>(0,1)); srcTri[1] = Point2f(meanMask.at<double>(1,0),meanMask.at<double>(1,1));
	//dstTri[0].x = 0.5*(keys[5*2]+keys[8*2]);dstTri[0].y = 0.5*(keys[11]+keys[17]);
	//dstTri[1].x = 0.5*(keys[11*2]+keys[14*2]);dstTri[1].y =0.5*(keys[11*2+1]+keys[14*2+1]);

	dstTri[0].x = 0.5*(keys[36]+keys[39]);	
	dstTri[0].y = 0.5*(keys[36+nLandmarks]+keys[39+nLandmarks]);
	dstTri[1].x = 0.5*(keys[42]+keys[45]);	
	dstTri[1].y =0.5*(keys[42+nLandmarks]+keys[45+nLandmarks]);

	Mat matTransform = Mat::zeros( 2, 3, CV_32F);
	Mat matRotate = Mat::zeros( 2, 3, CV_32F);
	float angle1 = cv::fastAtan2(srcTri[1].y-srcTri[0].y,srcTri[1].x-srcTri[0].x);
	float angle2 = cv::fastAtan2(dstTri[1].y-dstTri[0].y,dstTri[1].x-dstTri[0].x);
	float angle = angle1-angle2;
	float len1 = sqrt(pow(srcTri[1].y-srcTri[0].y,2)+pow(srcTri[1].x-srcTri[0].x,2));
	float len2 = sqrt(pow(dstTri[1].y-dstTri[0].y,2)+pow(dstTri[1].x-dstTri[0].x,2));
	float scale = len2/len1;
	Point2f center = srcTri[0];
	matRotate = cv::getRotationMatrix2D(center,angle,scale);
	matTransform.at<float>(0,0) = 1;
	matTransform.at<float>(1,1) = 1;
	matTransform.at<float>(0,2) = dstTri[0].x-srcTri[0].x;
	matTransform.at<float>(1,2) = dstTri[0].y-srcTri[0].y;

	Mat m_temp,m_pat;
	warpAffine( mask, m_temp, matRotate, mask.size() );
	warpAffine( m_temp, affinemask, matTransform, affinemask.size() );
	warpAffine( maskPat, m_pat, matRotate, maskPat.size() );
	warpAffine( m_pat, affinemaskPat, matTransform, affinemaskPat.size() );

}
void CGlasses::getHairTemplate(Mat oriImg,Mat maskPat,Mat mask,Mat& affinemaskPat,Mat& affinemask,float* keys)
{
	affinemask= Mat::zeros( oriImg.rows, oriImg.cols, oriImg.type());
	affinemaskPat = Mat::zeros( oriImg.rows, oriImg.cols, oriImg.type());

	vector<Point2d> affinePts;
	Mat m = meanMask;
	//vector<Point2d> vecMeanMask = Mat_<Point2d>(m);  
	vector<Point2f> srcTri(2);
	vector<Point2f> dstTri(2);
	//srcTri[0] = Point2f(vecMeanMask[0]); srcTri[1] = Point2f(vecMeanMask[1]);
	srcTri[0] = Point2f(meanMask.at<double>(0,0),meanMask.at<double>(0,1)); 
	srcTri[1] = Point2f(meanMask.at<double>(1,0),meanMask.at<double>(1,1));
	//dstTri[0].x = keys[5*2];dstTri[0].y = keys[11];
	//dstTri[1].x = keys[14*2];dstTri[1].y =keys[14*2+1];

	dstTri[0].x = 0.5*(keys[36]+keys[39]);	
	dstTri[0].y = 0.5*(keys[36+nLandmarks]+keys[39+nLandmarks]);
	dstTri[1].x = 0.5*(keys[42]+keys[45]);	
	dstTri[1].y =0.5*(keys[42+nLandmarks]+keys[45+nLandmarks]);

	Mat matTransform = Mat::zeros( 2, 3, CV_32F);
	Mat matRotate = Mat::zeros( 2, 3, CV_32F);
	float angle1 = cv::fastAtan2(srcTri[1].y-srcTri[0].y,srcTri[1].x-srcTri[0].x);
	float angle2 = cv::fastAtan2(dstTri[1].y-dstTri[0].y,dstTri[1].x-dstTri[0].x);
	float angle = angle1-angle2;
	float len1 = sqrt(pow(srcTri[1].y-srcTri[0].y,2)+pow(srcTri[1].x-srcTri[0].x,2));
	float len2 = sqrt(pow(dstTri[1].y-dstTri[0].y,2)+pow(dstTri[1].x-dstTri[0].x,2));
	float scale = len2/len1;
	Point2f center = srcTri[0];
	matRotate = cv::getRotationMatrix2D(center,angle,scale);
	matTransform.at<float>(0,0) = 1;
	matTransform.at<float>(1,1) = 1;
	matTransform.at<float>(0,2) = dstTri[0].x-srcTri[0].x;
	matTransform.at<float>(1,2) = dstTri[0].y-srcTri[0].y;

	Mat m_temp,m_pat;
	warpAffine( mask, m_temp, matRotate, mask.size() );
	warpAffine( m_temp, affinemask, matTransform, affinemask.size() );
	warpAffine( maskPat, m_pat, matRotate, maskPat.size() );
	warpAffine( m_pat, affinemaskPat, matTransform, affinemaskPat.size() );
}
int CGlasses::ProcessMerge(Mat img,Mat alpha)
{

	Mat mask8U, maskLab,mask_1,mask_2,mask_0;
	vector<Mat> vecLabMask;

	maskLab = img;
	vecLabMask.resize(maskLab.channels());
	split(maskLab, vecLabMask);
	vecLabMask[0].convertTo(mask_0,CV_64F);
	vecLabMask[1].convertTo(mask_1,CV_64F);
	vecLabMask[2].convertTo(mask_2,CV_64F);

	Mat img0,img1, img2,img3, maskShape_f;
	faceVecBGRImg[0].convertTo(img0,CV_64F);
	faceVecBGRImg[1].convertTo(img1,CV_64F);
	faceVecBGRImg[2].convertTo(img2,CV_64F);
	alpha.convertTo(maskShape_f, CV_64F);

	maskShape_f = maskShape_f/255.0;
	Mat m1,m2;

	m1=img1.mul(1-maskShape_f);
	m2 = mask_1.mul(maskShape_f);//ic{:,:,1}
	img1 = m1 + m2;
	img1.convertTo(faceVecBGRImg[1], CV_8U);

	m1=img2.mul(1-maskShape_f);
	m2 = mask_2.mul(maskShape_f);//ic{:,:,1}
	img2 = m1 + m2;
	img2.convertTo(faceVecBGRImg[2], CV_8U);

	m1=img0.mul(1-maskShape_f);
	m2 = mask_0.mul(maskShape_f);//ic{:,:,1}
	img0 = m1 + m2;
	img0.convertTo(faceVecBGRImg[0], CV_8U);
	return 1;
}

void CGlasses::ProcessGlasses(Mat& outImg,Mat faceImg,Mat maskPat,Mat mask,float* keys, int Landmarks)
{
	nLandmarks = Landmarks;
	oriImg = faceImg;
	
	faceVecBGRImg.resize(oriImg.channels());
	split(oriImg, faceVecBGRImg);

	Mat affineglasses;
	Mat affineAlpha;
	getGlassesTemplate(oriImg,maskPat,mask,affineglasses,affineAlpha,keys);
	ProcessMerge(affineglasses,affineAlpha);

	outImg = Mat::zeros(oriImg.size(), oriImg.type());
	merge(faceVecBGRImg, outImg);

}
void CGlasses::ProcessHair(Mat& outImg,Mat faceImg,Mat maskPat,Mat mask,float* keys, int Landmarks)
{
	nLandmarks = Landmarks;
	oriImg = faceImg;
	faceVecBGRImg.resize(oriImg.channels());
	split(oriImg, faceVecBGRImg);

	Mat affineHair;
	Mat affineAlpha;
	getHairTemplate(oriImg,maskPat,mask,affineHair,affineAlpha,keys);
	ProcessMerge(affineHair,affineAlpha);

	outImg = Mat::zeros(oriImg.size(), oriImg.type());
	merge(faceVecBGRImg, outImg);

}
void CGlasses::ProcessWaterMark(Mat& outImg,Mat faceImg,Mat mask)
{
	oriImg = faceImg;
	
	faceVecBGRImg.resize(oriImg.channels());
	split(oriImg, faceVecBGRImg);
	Mat affineAlpha;
	cv::resize(mask,affineAlpha,oriImg.size());
	cv::Mat black(oriImg.size(),oriImg.type(),cv::Scalar(128,128,128));
	ProcessMerge(black,affineAlpha);

	outImg = Mat::zeros(oriImg.size(), oriImg.type());
	merge(faceVecBGRImg, outImg);
}