#!/usr/bin/env sh

EXAMPLE=/home/lvjiangjing/caffe_exp/Data_aug/affine_p_160
CAFFE_PATH=/home/lvjiangjing/caffe_exp/caffe_code/key_point_multi_gpu_rand
cd $CAFFE_PATH

GLOG_logtostderr=0 GLOG_log_dir=$EXAMPLE/log ./build/tools/caffe train --solver=$EXAMPLE/quick_solver.prototxt --gpus=1 --snapshot=/home/lvjiangjing/caffe_exp/Data_aug/affine_p_160/models/_iter_700000.solverstate
#--weights=/home/lvjiangjing/caffe_exp/misalignment_exp/init/modles/_iter_249441.caffemodel 

#
