#pragma once
#ifndef _ALIGNEDFACE_H_
#define _ALIGNEDFACE_H_



#include <vector>
#include "opencv/cv.h"
#include "opencv/cxcore.h"

#define ALIGN_FACE_SIZE			160

#define ALIGN_EYE_RATIO_X		0.2f
#define ALIGN_EYE_RATIO_Y		0.25f
#define ALIGN_MOUTH_RATIO_X		0.5f
#define ALIGN_MOUTN_RATIO_Y		0.9f
#define RIGIDROT_WIDTH_RATIO	1.5f
#define RIGIDROT_UPPER_RATIO	1.2f
#define RIGIDROT_LOWER_RATIO	0.5f


/*
#define ALIGN_EYE_RATIO_X		0.30f
#define ALIGN_EYE_RATIO_Y		0.40f
#define ALIGN_MOUTH_RATIO_X		0.5f
#define ALIGN_MOUTN_RATIO_Y		0.75f
#define RIGIDROT_WIDTH_RATIO	1.5f
#define RIGIDROT_UPPER_RATIO	1.2f
#define RIGIDROT_LOWER_RATIO	0.5f
*/
#define NEW_ALIGN_EYE_RATIO_X		0.30f
#define NEW_ALIGN_EYE_RATIO_Y		0.40f
#define NEW_ALIGN_MOUTH_RATIO_X		0.5f
#define NEW_ALIGN_MOUTN_RATIO_Y		0.75f

#define MASK_START_RATIO_Y      0.30f
#define MASK_END_RATIO_Y        0.50f

#define nLandmarks  68
//#ifdef	ALIGNEDFACEDLL
//#define ALIGNEDFACEDLL __declspec(dllexport)
//#else
//#define ALIGNEDFACEDLL __declspec(dllimport)
//#endif


// we need a definition of CAlignModel
class CAlignModel
{
	public:
		cv::Point2f eyeRatio;
		cv::Point2f mRatio;
		int nFaceSize;
		double dWidthRatio;
		double dUpperRatio;
		double dLowerRatio;

		CAlignModel() { eyeRatio.x = ALIGN_EYE_RATIO_X; eyeRatio.y = ALIGN_EYE_RATIO_Y; 
								mRatio.x = ALIGN_MOUTH_RATIO_X; mRatio.y = ALIGN_MOUTN_RATIO_Y; 
								dWidthRatio = RIGIDROT_WIDTH_RATIO; dUpperRatio = RIGIDROT_UPPER_RATIO;
								dLowerRatio = RIGIDROT_LOWER_RATIO;
								nFaceSize = ALIGN_FACE_SIZE; }
		CAlignModel(cv::Point2f t_eyeRatio, cv::Point2f t_mRatio,int t_nFaceSize) { eyeRatio = t_eyeRatio; 
								mRatio = t_mRatio; nFaceSize = t_nFaceSize; }
		CAlignModel(double widthRatio, double upperRatio, double lowerRatio, int faceSize) 
			{ dWidthRatio = widthRatio; dUpperRatio = upperRatio; dLowerRatio = lowerRatio; nFaceSize = faceSize; }
};


class CAlignedFace
{

    public:    //
		CAlignedFace();
        CAlignedFace( CAlignModel * pAlignModel );
        void Init( CAlignModel * pAlignModel );
		void LoadModel(const char *filepath);

		cv::Mat & GetAlignedFaceImage() { return m_mAlignedFaceImage; }

		cv::Mat & Align(cv::Mat& faceImg, cv::Point m_apKeyPoints[3], int width, int height);
		/************** ljj  beg *************/
		// add Linear align 
		cv::Mat LinearAlign(cv::Mat& faceImg, cv::Point m_apKeyPoints[3]);
		// add piecewise align
		void DrawLabelsMask(cv::Mat& imgLabel,std::vector<cv::Point>& points);
		void PiecewiseWarpAffine(cv::Mat& img,std::vector<cv::Point>& s_0,cv::Mat& dst);
		void CalcCoeffs(std::vector<cv::Point>& s_0,std::vector<cv::Point>& s_1, cv::Mat& Coeffs);
		cv::Mat PiecewiseAlign(cv::Mat& faceImg, float *key_ptrs, bool landmark_perturbation, int perturb_range);

		/************** ljj end ********/
		cv::Mat & Mask();

		cv::Mat & RigidRotate(cv::Point apKeyPoints[3], double &dScore, const cv::Mat *mOrgImg, std::vector<cv::Point> *apDstKeyPoints=NULL, bool bKeepOrgSize=false, float fEnlargeRate=1.0);
		cv::Mat & RigidRotate(cv::Mat& faceImg, cv::Point m_apKeyPoints[3], double & dScore, const cv::Mat* mOrgImg, double dScale);
		

		cv::Mat & GetTransform() { return m_mTransform; }
		cv::Point2f PointMean(float keyPoints[],int startIdx,int endIdx);
		cv::Mat m_mTransform;

    private:
        CAlignModel * m_pcAlignModel;
		cv::Mat m_mAlignedFaceImage;
		cv::Mat masked_AlignedFaceImage;
		//cv::Mat m_mTransform;
		std::vector<cv::Mat> faceVecBGRImg;
		// add ljj
		cv::Mat ad_kp;
		cv::Mat dstLabelsMask;
		std::vector<cv::Point> s_1;
		std::vector<std::vector<unsigned int > > triangles;
};

#endif
