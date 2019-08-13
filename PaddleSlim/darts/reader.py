# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from PIL import ImageOps
import os
import math
import random
import _pickle as cPickle
import numpy as np
from PIL import Image
import paddle

IMAGE_SIZE = 32
IMAGE_DEPTH = 3
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


def preprocess(sample, is_training, args):
    image_array = sample.reshape(IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE)
    rgb_array = np.transpose(image_array, (1, 2, 0))
    img = Image.fromarray(rgb_array, 'RGB')

    if is_training:
        # pad, ramdom crop, random_flip_left_right
        img = ImageOps.expand(img, (4, 4, 4, 4), fill=0)
        left_top = np.random.randint(9, size=2)
        img = img.crop((left_top[0], left_top[1], left_top[0] + IMAGE_SIZE,
                        left_top[1] + IMAGE_SIZE))
        if np.random.randint(2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img).astype(np.float32)

    img_float = img / 255.0
    img = (img_float - CIFAR_MEAN) / CIFAR_STD

    if is_training and args.cutout:
        center = np.random.randint(IMAGE_SIZE, size=2)
        offset_width = max(0, center[0] - args.cutout_length // 2)
        offset_height = max(0, center[1] - args.cutout_length // 2)
        target_width = min(center[0] + args.cutout_length // 2, IMAGE_SIZE)
        target_height = min(center[1] + args.cutout_length // 2, IMAGE_SIZE)

        for i in range(offset_height, target_height):
            for j in range(offset_width, target_width):
                img[i][j][:] = 0.0

    img = np.transpose(img, (2, 0, 1))
    return img


def reader_generator(train_datasets, val_datasets, batch_size, is_training,
                     args):
    def read_batch(datasets, args):
        for im, label in datasets:
            im = preprocess(im, is_training, args)
            yield im, [int(label)]

    def reader():
        train_batch_data = []
        train_batch_label = []
        val_batch_data = []
        val_batch_label = []
        for train, val in zip(
                read_batch(train_datasets, args),
                read_batch(val_datasets, args)):
            train_batch_data.append(train[0])
            train_batch_label.append(train[1])
            val_batch_data.append(val[0])
            val_batch_label.append(val[1])
            if len(train_batch_data) == batch_size:
                train_batch_data = np.array(train_batch_data, dtype='float32')
                train_batch_label = np.array(train_batch_label, dtype='int64')
                val_batch_data = np.array(val_batch_data, dtype='float32')
                val_batch_label = np.array(val_batch_label, dtype='int64')
                batch_out = [
                    train_batch_data, train_batch_label, val_batch_data,
                    val_batch_label
                ]
                yield batch_out
                train_batch_data = []
                train_batch_label = []
                val_batch_data = []
                val_batch_label = []

    return reader


def train_val(args, batch_size, train_portion=1, is_shuffle=True):
    """
    CIFAR-10 training set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Training reader creator
    :rtype: callable
    """
    files = os.listdir(args.data)
    names = [each_item for each_item in files if 'data_batch' in each_item]
    names.sort()
    datasets = []
    for name in names:
        print("Reading file " + name)
        batch = cPickle.load(
            open(os.path.join(args.data, name), 'rb'), encoding='iso-8859-1')
        data = batch['data']
        labels = batch.get('labels', batch.get('fine_labels', None))
        assert labels is not None
        dataset = zip(data, labels)
        datasets.extend(dataset)
    if is_shuffle:
        random.shuffle(datasets)
    split_point = int(np.floor(train_portion * len(datasets)))
    train_datasets = datasets[:split_point]
    val_datasets = datasets[split_point:]
    readers = []
    n = int(math.ceil(len(train_datasets) // args.num_workers)
            ) if args.use_multiprocess_reader else len(train_datasets)
    train_datasets_lists = [
        train_datasets[i:i + n] for i in range(0, len(train_datasets), n)
    ]
    val_datasets_lists = [
        val_datasets[i:i + n] for i in range(0, len(val_datasets), n)
    ]
    for data_list in zip(train_datasets_lists, val_datasets_lists):
        readers.append(
            reader_generator(data_list[0], data_list[1], batch_size, True,
                             args))
    if args.use_multiprocess_reader:
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return readers[0]
