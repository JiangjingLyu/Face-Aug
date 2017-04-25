#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  //float temp_keys[nLandmarks*2]; 
  filenames_imginfos.clear();
  while (infile >> filename) {
    //LOG(INFO) << filename;
    //cv::Mat cv_img = imread (root_folder+filename,0);
    Imginfo imginfo;
    float read_f;
    for(int i=0;i<nLandmarks*2;++i){
      infile>>read_f;
      imginfo.keys[i]=read_f;
    }
   

    for(int i=0;i<3;++i){
      infile>>read_f;
      imginfo.angles[i]=read_f; 
    }
    infile>>read_f;
    imginfo.glasses_score=read_f;
    filenames_imginfos[filename]=imginfo;

    infile>>label;
    lines_.push_back(std::make_pair(filename, label));
  }
  assert(filenames_imginfos.size()==lines_.size());


  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  
  cv::Mat cv_img;
  cv::Mat cv_img_tmp = imread(root_folder+lines_[lines_id_].first, cv_read_flag);
  alignedface.Init(&alignmodel);
  string model_name = "./include/caffe/models/piece_align_68.bin";//"/home/lvjiangjing/caffe_exp/caffe_code/key_point_multi_gpu/include/caffe/models/piece_align_68.bin";
  alignedface.LoadModel(model_name.c_str());
  // Read an image, and use it to initialize the top blob.
  //cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, new_height, new_width, is_color);


   Imginfo& imginfo= filenames_imginfos[lines_[lines_id_].first];
   float* keys=imginfo.keys;
   cv::Mat aligned_Img;
   cv::Point kp_for_alignment[3];
   int key_idx[6]={36,41,42,47,60,67};
   for(unsigned int i=0;i<3;++i)
   {
      kp_for_alignment[i]=alignedface.PointMean(keys,key_idx[2*i],key_idx[2*i+1]);
   }
  

  /***************** align *******************/
   double score;
   cv::Mat aligned_Img_color =  alignedface.RigidRotate(cv_img_tmp,kp_for_alignment,score,0,1); //Affine transformation
   

   cv::resize(aligned_Img_color, cv_img, cv::Size(new_width, new_height));
   //imshow("init_img", cv_img);
   //cv::waitKey(0);
      
   /*if(!is_color)
   {
     cvtColor(aligned_Img_color,cv_img, CV_BGR2GRAY);    
   }
   else
   {
    cv_img = aligned_Img_color;
   }*/

  
  

  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  //LOG(INFO) << "channels = "<<cv_img.channels()<<"cv_img.rows = "<<cv_img.rows << "cv_img.cols = "<<cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) 
  {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
      this->prefetch_[i].data_.Reshape(batch_size, channels,
          crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } 
  else 
  {
    top[0]->Reshape(batch_size, channels, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
      this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);

    //;LOG(INFO) << "channels = "<<cv_img.channels()<<"cv_img.rows = "<<cv_img.rows << "cv_img.cols = "<<cv_img.cols;
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
  {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  const int glasses_model_count= image_data_param.glasses_model_count();
  const int hair_model_count = image_data_param.hair_model_count();
  const int face_align_type = image_data_param.face_align_type();
  const bool landmark_perturbation = image_data_param.landmark_perturbation();
  const bool add_glass = image_data_param.add_glass();
  const int perturb_range = image_data_param.perturb_range();
  const bool add_hair = image_data_param.add_hair();

  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  // Reshape on single input batches for inputs of varying dimension.
  if (batch_size == 1 && crop_size == 0 && new_height == 0 && new_width == 0) {
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    batch->data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
    this->transformed_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
  }

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  cv::Point kp_for_alignment[3];
  int key_idx[6]={36,41,42,47,60,67};
  //float tmp_keys[nLandmarks*2];

  Mat glassesImg;
  Mat hairImg;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    //cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,new_height, new_width, is_color);
    cv::Mat cv_img = imread (root_folder+lines_[lines_id_].first,cv_read_flag);


    if(cv_img.data==NULL)
    {
      item_id--; 
      lines_id_++;
      LOG(INFO)<<"could not load image"<<lines_[lines_id_].first<<"skip it"; 
      if (lines_id_ >= lines_size) 
      {
         // We have reached the end. Restart from the first.
         DLOG(INFO) << "Restarting data prefetching from start.";
         lines_id_ = 0;
         if (this->layer_param_.image_data_param().shuffle()) 
         {
           ShuffleImages();
         }
      }
      continue;
    }


    Imginfo& imginfo= filenames_imginfos[lines_[lines_id_].first];
    float* keys=imginfo.keys;
    //for(unsigned int i=0;i<nLandmarks*2;++i)
    //{
     // tmp_keys[i]=keys[i];  
    //}

    for(unsigned int i=0;i<3;++i)
    {
      kp_for_alignment[i]=alignedface.PointMean(keys,key_idx[2*i],key_idx[2*i+1]);
    }

    if(add_glass)
    {
      if(rand()%4==0&&imginfo.glasses_score>0.4&&fabs(imginfo.angles[0])<25&&fabs(imginfo.angles[1]<25))
      {
         const string& glasses_img_prefix = image_data_param.glasses_img();
         const string& glasses_alpha_img_prefix = image_data_param.glasses_alpha_img();
         int model_idx=rand()%glasses_model_count;
         char temp[1024];
         sprintf(temp,"/glass%d.jpg",model_idx);
         string glasses_img=glasses_img_prefix;
         glasses_img.append(temp);

         sprintf(temp,"/glass%d_alpha.jpg",model_idx);
         string glasses_alpha_img=glasses_alpha_img_prefix;
         glasses_alpha_img.append(temp);

         glasses = cv::imread(glasses_img.c_str(),1);
         glasses_alpha = cv::imread(glasses_alpha_img.c_str(),0);
         assert(glasses.data()!=0 && glasses_alpha.data()!= 0);

         //LOG(INFO) << "Starting glasses processed: "<< glasses_img;
         cglasses.ProcessGlasses(glassesImg,cv_img,glasses,glasses_alpha,keys, nLandmarks);
         //LOG(INFO) << "Ending glasses processed.";
         cv_img=glassesImg;
      }  
    }
    if(add_hair)
    {
      if(rand()%4==0 &&fabs(imginfo.angles[0])<25&&fabs(imginfo.angles[1]<25))
      {

         const string& hair_img_prefix = image_data_param.hair_img();
         const string& hair_alpha_img_prefix = image_data_param.hair_alpha_img();
         int model_idx=rand()%hair_model_count;
         char temp[1024];
         sprintf(temp,"/hair%d.jpg",model_idx);
         string hair_img=hair_img_prefix;
         hair_img.append(temp);

         sprintf(temp,"/hair%d_alpha.jpg",model_idx);
         string hair_alpha_img=hair_alpha_img_prefix;
         hair_alpha_img.append(temp);

         hair = cv::imread(hair_img.c_str(),1);
         hair_alpha = cv::imread(hair_alpha_img.c_str(),0);
         assert(hair.data()!=0 && hair_alpha.data()!= 0);

         //LOG(INFO) << "Starting hair processed: "<< hair_alpha_img;
         cglasses.ProcessHair(hairImg,cv_img,hair,hair_alpha,keys, nLandmarks);
         //LOG(INFO) << "Ending glasses processed.";
         cv_img=hairImg;
      }  
    }    

   // LOG(INFO) << "channels = "<<cv_img.channels()<<"cv_img.rows = "<<cv_img.rows << "cv_img.cols = "<<cv_img.cols;
    /*for(int j=0;j<nLandmarks;j++)
    {
      printf("(%f,%f)\n",keys[j],keys[j+nLandmarks]);
      cv::circle(cv_img,cv::Point(cvRound(keys[j]),cvRound(keys[j+nLandmarks])),1,cv::Scalar(0,0,255),2);
    }
    imshow("src_img", cv_img);
    cv::waitKey(0);*/
    
    /***************** align *******************/
    int sel_align;
    if(face_align_type == 0)
      {
        sel_align = rand()%3+1;
      }
    else
    {
      sel_align = face_align_type;
    }

    if(landmark_perturbation)
    {
      for(unsigned int i=0;i<3;++i)
      {

        kp_for_alignment[i].x=kp_for_alignment[i].x + rand()%(perturb_range*2)-perturb_range;
        kp_for_alignment[i].y=kp_for_alignment[i].y + rand()%(perturb_range*2)-perturb_range;
        kp_for_alignment[i].x = max(0,kp_for_alignment[i].x);
        kp_for_alignment[i].x = min(cv_img.cols,kp_for_alignment[i].x);
        kp_for_alignment[i].y = max(kp_for_alignment[i].y,0);
        kp_for_alignment[i].y = min(cv_img.rows,kp_for_alignment[i].y); 
      }
    }

    double score;
    cv::Mat aligned_Img_color;
    switch(sel_align)
    {
     case 1:
       aligned_Img_color =  alignedface.RigidRotate(cv_img,kp_for_alignment,score,0,1); //Affine transformation
       break;
     case 2:
       aligned_Img_color =  alignedface.LinearAlign(cv_img,kp_for_alignment);
       break;
     case 3:
       aligned_Img_color =  alignedface.PiecewiseAlign(cv_img,keys,landmark_perturbation,perturb_range);
       break;
    }
     

     cv::resize(aligned_Img_color, cv_img, cv::Size(new_width, new_height));
     //imshow("img", cv_img);//
     //cv::waitKey(0);//


    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe