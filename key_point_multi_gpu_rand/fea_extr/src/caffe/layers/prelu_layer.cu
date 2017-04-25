#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void PReLUForward(const int n, const Dtype* in, Dtype* out, const Dtype* mask) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    out[index] = in[index] > 0 ? in[index] : in[index] /mask[index];
  }
}

template <typename Dtype>
__global__ void PReLUForward(const int n, const Dtype* in, Dtype* out, Dtype negative_slope) 
{
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}


template <typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)  
  if (this->phase_ == TRAIN)
  { 
    Dtype r;
    caffe_rng_uniform(1, Dtype(3), Dtype(8), &r);
    rand_negative_slope_ = 1/(r);
    //LOG(INFO) << "rand_negative_slope_ = " << rand_negative_slope_;

    PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data, rand_negative_slope_);
    CUDA_POST_KERNEL_CHECK;
  }
  else
  {
    Dtype slope_ = 0.18181818181;
    PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data, slope_);
    CUDA_POST_KERNEL_CHECK;
  }


  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void PReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype * mask) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0) + (in_data[index] <= 0) / mask[index]);
  }
}

template <typename Dtype>
__global__ void PReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}


template <typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[0]) 
  {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    

    if (this->phase_ == TRAIN)
    {    
      PReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data, bottom_diff, rand_negative_slope_);
      CUDA_POST_KERNEL_CHECK;
    }
    else
    {
      Dtype slope_ = 0.18181818181;
      PReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data, bottom_diff, slope_);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);


}  // namespace caffe
