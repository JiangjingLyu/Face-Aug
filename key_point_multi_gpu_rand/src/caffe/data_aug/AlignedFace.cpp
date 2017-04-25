
#define ALIGNEDFACEDLL

#include <vector>
#include "caffe/data_aug/AlignedFace.h"
#include "opencv/highgui.h"


using namespace std;


CAlignedFace::CAlignedFace()
{
	faceVecBGRImg.reserve(3);
	int S = 200-1;
	int H = cvRound(S/2)-1;
	ad_kp =(cv::Mat_<double>(8,2)<<0,0,0,H,0,S,H,S,S,S,S,H,S,0,H,0);
	ad_kp = ad_kp.t();

	dstLabelsMask = cv::Mat::zeros(200,200,CV_32S);
}
CAlignedFace::CAlignedFace( CAlignModel * pAlignModel )
{
	m_pcAlignModel = pAlignModel;
	faceVecBGRImg.reserve(3);

	int S = 200-1;
	int H = cvRound(S/2)-1;
	ad_kp =(cv::Mat_<double>(8,2)<<0,0,0,H,0,S,H,S,S,S,S,H,S,0,H,0);
	ad_kp = ad_kp.t();

	dstLabelsMask = cv::Mat::zeros(200,200,CV_32S);
}

        
void CAlignedFace::Init( CAlignModel * pAlignModel )  // nFaceID: [0 .. nFaceNum-1]
{
	m_pcAlignModel = pAlignModel;
	faceVecBGRImg.reserve(3);
}

void CAlignedFace::LoadModel(const char *filepath)
{
	FILE *fpb;
	if((fpb = fopen(filepath,"rb"))==NULL)
	{
		printf("Cant open the file\n" );
		return;
	}

	cv::Mat std_kpt(76,2,CV_32F);
	cv::Mat tri(87,3,CV_32F);
	fread((char *)(std_kpt.data),sizeof(float)*76*2,1,fpb);
	fread((char *)(tri.data),sizeof(float)*87*3,1,fpb);
	fclose(fpb);

	
	std::vector<unsigned int > tmp;
	int j;
	for(j=0;j<76;j++)
	{
		s_1.push_back(cv::Point(cvRound(std_kpt.at<float>(j,0)),cvRound(std_kpt.at<float>(j,1))));

	}
	for(j=0;j<87;j++)
	{
		tmp.empty();
		tmp.clear();
		tmp.push_back(tri.at<float>(j,0));
		tmp.push_back(tri.at<float>(j,1));
		tmp.push_back(tri.at<float>(j,2));
		triangles.push_back(tmp);

	}

	DrawLabelsMask(dstLabelsMask,s_1);
}


cv::Mat & CAlignedFace::Align(cv::Mat& faceImg, cv::Point m_apKeyPoints[3], int width, int height)
{
	cv::Point2f srcKeyPoint[3];
	cv::Point2f dstKeyPoint[3];
	cv::Mat map_matrix;
	cv::Size dsize;
	dsize.height = height;//m_pcAlignModel->nFaceSize;
	dsize.width = width;//m_pcAlignModel->nFaceSize;

	
	dstKeyPoint[0].x = (m_pcAlignModel->nFaceSize * m_pcAlignModel->eyeRatio.x);
	dstKeyPoint[0].y = (m_pcAlignModel->nFaceSize * m_pcAlignModel->eyeRatio.y);
	dstKeyPoint[1].x = (m_pcAlignModel->nFaceSize * (1-m_pcAlignModel->eyeRatio.x));
	dstKeyPoint[1].y = dstKeyPoint[0].y;
	dstKeyPoint[2].x = (m_pcAlignModel->nFaceSize * m_pcAlignModel->mRatio.x);
	dstKeyPoint[2].y = (m_pcAlignModel->nFaceSize * m_pcAlignModel->mRatio.y);

	/*dstKeyPoint[0].x = (width * m_pcAlignModel->eyeRatio.x);
	dstKeyPoint[0].y = (height * m_pcAlignModel->eyeRatio.y);
	dstKeyPoint[1].x = (width * (1-m_pcAlignModel->eyeRatio.x));
	dstKeyPoint[1].y = dstKeyPoint[0].y;
	dstKeyPoint[2].x = (width * m_pcAlignModel->mRatio.x);
	dstKeyPoint[2].y = (height * m_pcAlignModel->mRatio.y);*/

	
	for(int j=0; j<3; j++ )
	{
		srcKeyPoint[j].x = float(m_apKeyPoints[j].x);
		srcKeyPoint[j].y = float(m_apKeyPoints[j].y);
	}

	map_matrix = cv::getAffineTransform(srcKeyPoint,dstKeyPoint);	

//	cv::Mat key=cv::Mat::zeros(3,1,CV_64F);
//int t = map_matrix.type();
//	key.at<double>(0,0) = srcKeyPoint[0].x;
//	key.at<double>(1,0) = srcKeyPoint[0].y;
//	key.at<double>(2,0) = 1;
//
//
//	cv::Mat dst = map_matrix * key;
//
//	float x=dst.at<double>(0,0);
//	float y=dst.at<double>(1,0);



	cv::warpAffine(faceImg,m_mAlignedFaceImage,map_matrix,dsize);
	m_mTransform = map_matrix.clone();
	imshow("warped", m_mAlignedFaceImage);
	cv::waitKey(0);
	return m_mAlignedFaceImage;
}

cv::Mat CAlignedFace::LinearAlign(cv::Mat& faceImg, cv::Point m_apKeyPoints[3])
{
	
	// Get the center between the 2 eyes.
	cv::Point2f eyesCenter = cv::Point2f( (m_apKeyPoints[0].x + m_apKeyPoints[1].x) * 0.5f, (m_apKeyPoints[0].y + m_apKeyPoints[1].y) * 0.5f );
	// Get the angle between the 2 eyes.
	double dy = abs(m_apKeyPoints[1].y - m_apKeyPoints[0].y);
	double dx = abs(m_apKeyPoints[1].x - m_apKeyPoints[0].x);
	double len = sqrt(dx*dx + dy*dy);
	double angle = atan2(dy, dx) * 180.0/CV_PI; // Convert from radians to degrees.

	// Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
	const double DESIRED_RIGHT_EYE_X = (1.0f - NEW_ALIGN_EYE_RATIO_X);
	// Get the amount we need to scale the image to be the desired fixed size we want.
	double desiredLen = (DESIRED_RIGHT_EYE_X - NEW_ALIGN_EYE_RATIO_X) * m_pcAlignModel->nFaceSize;
	double scale = desiredLen / len;
	// Get the transformation matrix for rotating and scaling the face to the desired angle & size.
	cv::Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
	// Shift the center of the eyes to be the desired center between the eyes.
	rot_mat.at<double>(0, 2) += m_pcAlignModel->nFaceSize * 0.5f - eyesCenter.x;
	rot_mat.at<double>(1, 2) += m_pcAlignModel->nFaceSize * NEW_ALIGN_EYE_RATIO_Y - eyesCenter.y;

	// Rotate and scale and translate the image to the desired angle & size & position!
	// Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
	cv::Mat warped = cv::Mat(m_pcAlignModel->nFaceSize, m_pcAlignModel->nFaceSize, faceImg.type(), cv::Scalar(128)); // Clear the output image to a default grey.
	cv::warpAffine(faceImg, warped, rot_mat, warped.size());
	/*imshow("warped", warped);
	cv::waitKey(0);*/
	return warped;
}


cv::Mat & CAlignedFace::Mask(){

	masked_AlignedFaceImage=m_mAlignedFaceImage;


	int start_y=ALIGN_FACE_SIZE*MASK_START_RATIO_Y;
	int end_y=ALIGN_FACE_SIZE*MASK_END_RATIO_Y;
	assert(start_y>=0&&end_y<ALIGN_FACE_SIZE&&start_y<=end_y);
	int height=end_y-start_y+1;

	cv::Mat roi(masked_AlignedFaceImage, cv::Rect(0,start_y,ALIGN_FACE_SIZE,height));
	roi = cv::Scalar(0,0,0);
	
	return masked_AlignedFaceImage;		
}


cv::Mat & CAlignedFace::RigidRotate(cv::Mat& faceImg, cv::Point m_apKeyPoints[3], double & dScore, const cv::Mat* mOrgImg, double dScale)
{
	cv::Point apKeyPoints[3];

	for (int i = 0; i < 3; i++)
		apKeyPoints[i] = m_apKeyPoints[i];


	if ( mOrgImg==NULL )
	{
		//return RigidRotate(apKeyPoints, dScore, &faceImg);
		return RigidRotate(apKeyPoints, dScore, &faceImg,NULL,false,dScale);
	}
	else
	{
		if ( mOrgImg->rows != int(faceImg.rows * dScale+0.5) || mOrgImg->cols != int(faceImg.cols * dScale + 0.5) )
			//Error("Original image should be of the same size as face detection image");

			if (fabs(dScale-1.0) > 1.0e-6)
			{
				for (int i = 0; i < 3; i++)
				{
					apKeyPoints[i].x = int(double(apKeyPoints[i].x) * dScale);
					apKeyPoints[i].y = int(double(apKeyPoints[i].y) * dScale);
				}
			}

			return RigidRotate(apKeyPoints, dScore, mOrgImg);

	}


}

cv::Mat & CAlignedFace::RigidRotate(cv::Point apKeyPoints[3], double & dScore, const cv::Mat* mOrgImg, vector<cv::Point> *apDstKeyPoints, bool bKeepOrgSize, float fEnlargeRate)
{
	cv::Point2f srcKeyPoint[3];
	cv::Point2f dstKeyPoint[3];
	cv::Mat map_matrix;
	cv::Size dsize(int(0.5+fEnlargeRate*m_pcAlignModel->nFaceSize), int(0.5+fEnlargeRate*m_pcAlignModel->nFaceSize));
	double dcorf_width = fEnlargeRate*m_pcAlignModel->dWidthRatio;
	double dcorf_upper = fEnlargeRate*m_pcAlignModel->dUpperRatio;
	double dcorf_lower = fEnlargeRate*m_pcAlignModel->dLowerRatio;

	cv::Point2f wVec, hVec;
	double angle, distEyeMouth;
	wVec = apKeyPoints[1] - apKeyPoints[0];
	angle = std::atan(wVec.y/wVec.x);
	
	hVec = cv::Point2f(-sin(angle),cos(angle));
	distEyeMouth = hVec.dot(apKeyPoints[2]-apKeyPoints[0]);
	hVec = distEyeMouth * hVec;
	
	srcKeyPoint[0] = cv::Point2f(apKeyPoints[0].x,apKeyPoints[0].y) - 0.5*dcorf_width*wVec - dcorf_upper*hVec;
	srcKeyPoint[1] = cv::Point2f(apKeyPoints[1].x,apKeyPoints[1].y) + 0.5*dcorf_width*wVec - dcorf_upper*hVec;
	srcKeyPoint[2] = srcKeyPoint[1] + (dcorf_upper+dcorf_lower+1)*hVec;

	
	cv::Point2f srcKeyPoint3 = srcKeyPoint[0] + (dcorf_upper+dcorf_lower+1)*hVec;
	if (srcKeyPoint[0].x < 0 || srcKeyPoint[0].y < 0 || 
		srcKeyPoint[1].x > mOrgImg->cols || srcKeyPoint[1].y < 0 ||
		srcKeyPoint[2].x > mOrgImg->cols || srcKeyPoint[2].y > mOrgImg->rows ||
		srcKeyPoint3.x < 0 || srcKeyPoint3.y > mOrgImg->rows)
		dScore = 0;
	else
		dScore = 1;

	dstKeyPoint[0] = cv::Point2f(0,0);
	dstKeyPoint[1] = cv::Point2f((float)dsize.width,0);
	dstKeyPoint[2] = cv::Point2f((float)dsize.width,(float)dsize.height);

	if (bKeepOrgSize)
	{
		float reverse_scale = min(srcKeyPoint[1].x - srcKeyPoint[0].x, srcKeyPoint[2].y - srcKeyPoint[1].y) / (float)dsize.width;
		if (reverse_scale < 1)
			reverse_scale = 1.0;
		dsize.width = int(dsize.width * reverse_scale + 0.5);
		dsize.height = int (dsize.height * reverse_scale + 0.5);
		dstKeyPoint[1].x = dsize.width;
		dstKeyPoint[2].x = dsize.width;
		dstKeyPoint[2].y = dsize.height;
	}

	// maybe need to check boundaries?
	map_matrix = cv::getAffineTransform(srcKeyPoint,dstKeyPoint);
	cv::warpAffine(*mOrgImg,m_mAlignedFaceImage,map_matrix,dsize);

	if (apDstKeyPoints)
	{
		double mx0, mx1, mx2, my0, my1, my2;
		mx0 = map_matrix.at<double>(0,0);
		mx1 = map_matrix.at<double>(0,1);
		mx2 = map_matrix.at<double>(0,2);
		my0 = map_matrix.at<double>(1,0);
		my1 = map_matrix.at<double>(1,1);
		my2 = map_matrix.at<double>(1,2);

		apDstKeyPoints->resize(3);
		(*apDstKeyPoints)[0].x = int(mx0*apKeyPoints[0].x + mx1*apKeyPoints[0].x + mx2 + 0.5);
		(*apDstKeyPoints)[0].y = int(my0*apKeyPoints[0].y + my1*apKeyPoints[0].y + my2 + 0.5);
		(*apDstKeyPoints)[1].x = int(mx0*apKeyPoints[1].x + mx1*apKeyPoints[1].x + mx2 + 0.5);
		(*apDstKeyPoints)[1].y = int(my0*apKeyPoints[1].y + my1*apKeyPoints[1].y + my2 + 0.5);
		(*apDstKeyPoints)[2].x = int(mx0*apKeyPoints[2].x + mx1*apKeyPoints[2].x + mx2 + 0.5);
		(*apDstKeyPoints)[2].y = int(my0*apKeyPoints[2].y + my1*apKeyPoints[2].y + my2 + 0.5);
	}

	map_matrix.copyTo(m_mTransform);
	return m_mAlignedFaceImage;
}

cv::Point2f CAlignedFace::PointMean(float keyPoints[],int startIdx,int endIdx)
{
	//int nLandmarks = 68;
	int meanLength = endIdx - startIdx + 1;
	cv::Point2f mpt;
	mpt.x = 0;
	mpt.y = 0;

	for (int j = startIdx;j<=endIdx;j++)
	{
		mpt.x += keyPoints[j]; 
		mpt.y += keyPoints[nLandmarks+j]; 
	}

	mpt.x = mpt.x/meanLength;
	mpt.y = mpt.y/meanLength;
	return mpt;
}

/************************    piece wise ****************************/
void CAlignedFace::DrawLabelsMask(cv::Mat& imgLabel,std::vector<cv::Point>& points)
{
	for(int i=0;i<triangles.size();i++)
	{
		cv::Point t[3];
		int ind1=triangles[i][0];
		int ind2=triangles[i][1];
		int ind3=triangles[i][2];
		t[0].x=cvRound(points[ind1].x);
		t[0].y=cvRound(points[ind1].y);
		t[1].x=cvRound(points[ind2].x);
		t[1].y=cvRound(points[ind2].y);
		t[2].x=cvRound(points[ind3].x);
		t[2].y=cvRound(points[ind3].y);
		cv::fillConvexPoly(imgLabel, t, 3, cv::Scalar_<int>((i+1)));

	}
}

void CAlignedFace::CalcCoeffs(std::vector<cv::Point>& s_0,std::vector<cv::Point>& s_1, cv::Mat& Coeffs)
{
	cv::Rect_<int> Bound_0;
	cv::Rect_<int> Bound_1;
	// ±ß½ç
	
	Bound_0=cv::boundingRect(cv::Mat(s_0));
	Bound_1=cv::boundingRect(cv::Mat(s_1));
	Coeffs=cv::Mat(triangles.size(),6,CV_64FC1);
	
	for(int i=0;i<triangles.size();i++)
	{
		int ind1=triangles[i][0];
		int ind2=triangles[i][1];
		int ind3=triangles[i][2];
		// 
		cv::Point2d t_0[3];
		t_0[0]=s_0[ind1]-Bound_0.tl(); // i
		t_0[1]=s_0[ind2]-Bound_0.tl(); // j
		t_0[2]=s_0[ind3]-Bound_0.tl(); // k
		//
		cv::Point2d t_1[3];
		t_1[0]=s_1[ind1]-Bound_1.tl(); // i
		t_1[1]=s_1[ind2]-Bound_1.tl(); // j
		t_1[2]=s_1[ind3]-Bound_1.tl(); // k

		double denom=(t_1[0].x * t_1[1].y + t_1[2].y * t_1[1].x - t_1[0].x * t_1[2].y - t_1[2].x * t_1[1].y - t_1[0].y * t_1[1].x + t_1[0].y * t_1[2].x);
		
		Coeffs.at<double>(i,0)= -(-t_1[2].y * t_0[1].x + t_1[2].y * t_0[0].x + t_1[1].y * t_0[2].x - t_1[1].y * t_0[0].x - t_1[0].y * t_0[2].x + t_1[0].y * t_0[1].x) / denom;
		Coeffs.at<double>(i,1)= -(t_1[2].x * t_0[1].x - t_1[2].x * t_0[0].x - t_1[1].x * t_0[2].x + t_1[1].x * t_0[0].x + t_1[0].x * t_0[2].x - t_1[0].x * t_0[1].x) / denom;
		Coeffs.at<double>(i,2)= -(t_1[2].x * t_1[1].y * t_0[0].x - t_1[2].x * t_1[0].y * t_0[1].x - t_1[1].x * t_1[2].y * t_0[0].x + t_1[1].x * t_1[0].y * t_0[2].x + t_1[0].x * t_1[2].y * t_0[1].x - t_1[0].x * t_1[1].y * t_0[2].x)/denom;
		Coeffs.at<double>(i,3)= -(t_1[1].y * t_0[2].y - t_1[0].y * t_0[2].y - t_1[2].y * t_0[1].y + t_1[2].y * t_0[0].y - t_0[0].y * t_1[1].y + t_0[1].y * t_1[0].y) / denom;
		Coeffs.at<double>(i,4)= -(-t_1[2].x * t_0[0].y + t_1[0].x * t_0[2].y + t_1[2].x * t_0[1].y - t_0[1].y * t_1[0].x - t_1[1].x * t_0[2].y + t_0[0].y * t_1[1].x) / denom;
		Coeffs.at<double>(i,5)= -(t_0[0].y * t_1[1].y * t_1[2].x - t_0[2].y * t_1[0].x * t_1[1].y - t_0[1].y * t_1[0].y * t_1[2].x + t_0[1].y * t_1[0].x * t_1[2].y + t_0[2].y * t_1[0].y * t_1[1].x - t_0[0].y * t_1[1].x * t_1[2].y) / denom;
	}
}

void CAlignedFace::PiecewiseWarpAffine(cv::Mat& img,std::vector<cv::Point>& s_0,cv::Mat& dst)
{
	cv::Rect_<int> Bound_0;
	cv::Rect_<int> Bound_1;


	Bound_0=cv::boundingRect(cv::Mat(s_0));
	Bound_1=cv::boundingRect(cv::Mat(s_1));	

	Bound_1.width=cvRound(Bound_1.width);
	Bound_1.height=cvRound(Bound_1.height);

	Bound_0.width=cvRound(Bound_0.width);
	Bound_0.height=cvRound(Bound_0.height);



	if(Bound_0.br().x>img.cols-1){Bound_0.width=(double)img.cols-1-Bound_0.x;}
	if(Bound_0.br().y>img.rows-1){Bound_0.height=(double)img.rows-1-Bound_0.y;}
	

	cv::Mat I_0=img(Bound_0);

	// 
	for(int i=0;i<s_1.size();i++)
	{
	s_1[i]-=Bound_1.tl();
	}

	// 
	if(Bound_1.x<0)
	{
		Bound_1.x=0;
	}
	
	if(Bound_1.y<0)
	{
		Bound_1.y=0;
	}

	if(Bound_1.br().x>dst.cols-1)
	{
		Bound_1.width=(double)dst.cols-1-Bound_1.x;
	}
	
	if(Bound_1.br().y>dst.rows-1)
	{
		Bound_1.height=(double)dst.rows-1-Bound_1.y;
	}


	cv::Mat I_1=dst(Bound_1);

	cv::Mat Coeffs;
	CalcCoeffs(s_0,s_1,Coeffs);
	
	
	
	for(int i=0;i<I_1.rows;i++)
	{
		cv::Point2d W(0,0);
		for(int j=0;j<I_1.cols;j++)
		{
			double x=j;
			double y=i;
			int Label=dstLabelsMask.at<int>(i,j)-1;
			if(Label!=(-1))
			{				
				W.x=Coeffs.at<double>(Label,0)*x+Coeffs.at<double>(Label,1)*y+Coeffs.at<double>(Label,2);
				W.y=Coeffs.at<double>(Label,3)*x+Coeffs.at<double>(Label,4)*y+Coeffs.at<double>(Label,5);
				if(cvRound(W.x)>0 && cvRound(W.x)<I_0.cols && cvRound(W.y)>0 && cvRound(W.y)<I_0.rows)
				{
					/*if(i==139 && j==86)
					{
						cout<<i<<' '<<j<<endl;
					}*/
					//cout<<i<<' '<<j<<endl;
					
					I_1.at<cv::Vec3b>(i,j)=I_0.at<cv::Vec3b>(cvRound(W.y),cvRound(W.x));
				}
			}
		}
	}
	//cv::GaussianBlur(I_1,I_1,cv::Size(3,3),0.5);	
}


cv::Mat CAlignedFace::PiecewiseAlign(cv::Mat& faceImg, float *key_ptrs, bool landmark_perturbation, int perturb_range)
{
	cv::Point pt[3];
	int j;
	pt[0] = PointMean(key_ptrs,36,41);
	pt[1] = PointMean(key_ptrs,42,47);
	pt[2] = PointMean(key_ptrs,60,67);
	double score;
	cv::Mat aligned_Img_color = RigidRotate(faceImg,pt,score,0,1.25);

	cv::Mat affkpt = cv::Mat::ones(3,nLandmarks,CV_64F);
	for (j = 0; j < nLandmarks; j++)
	{
		affkpt.at<double>(0,j) = key_ptrs[j];
		affkpt.at<double>(1,j) = key_ptrs[j+nLandmarks];
	}
	cv::Mat new_kpt = m_mTransform*affkpt;

	cv::Mat combine;
	hconcat(new_kpt,ad_kp,combine);
	std::vector<cv::Point> s_0;
	for(j=0;j<76;j++)
	{
		int x,y;
		x =  cvRound(combine.at<double>(0,j));
		y = cvRound(combine.at<double>(1,j));
		if(landmark_perturbation && j< nLandmarks)
		{
			x = x+ rand()%(perturb_range*2)-perturb_range;
			y = y+ rand()%(perturb_range*2)-perturb_range;
		}

		x = max(0,x);
		x = min(aligned_Img_color.cols,x);
		y = max(y,0);
		y = min(aligned_Img_color.rows,y);
		s_0.push_back(cv::Point(x,y));
	}

	//imshow("pieceimg", aligned_Img_color);
    //cv::waitKey(0);


	cv::Mat dst=cv::Mat(aligned_Img_color.cols, aligned_Img_color.rows, aligned_Img_color.type(), cv::Scalar(0));//aligned_Img_color.clone();
	PiecewiseWarpAffine(aligned_Img_color,s_0,dst);
	cv::Rect rect(20,20,160,160);
	cv::Mat piece_img = dst(rect).clone();
	//imshow("piece_img",piece_img);
	//cv::waitKey(0);
	return piece_img;
}