#!/usr/bin/env bash

# download pretrain model
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
ResNet50="ResNet50_pretrained.tar"
pretrain_dir='../pretrain'

if [ ! -d ${pretrain_dir} ]; then
  mkdir ${pretrain_dir}
fi

cd ${pretrain_dir}

if [ ! -f ${ResNet50} ]; then
    wget ${root_url}/${ResNet50}
    tar xf ${ResNet50}
fi

cd -

# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0

# for distillation
#-----------------
export CUDA_VISIBLE_DEVICES=2

python -u compress.py \
--model "MobileNet" \
--teacher_model "ResNet50" \
--teacher_pretrained_model ../pretrain/ResNet50_pretrained \
#> mobilenet_v1.log 2>&1 &
#tailf mobilenet_v1.log
