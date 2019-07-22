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

import os
import math
import random
import numpy as np
from PIL import Image


def _pre_process(x, label):
    cutout_length = args.cutout_length
    x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
    x = tf.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    if cutout_length is not None:
        mask = tf.ones([cutout_length, cutout_length], dtype=tf.int32)
        start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
        mask = tf.pad(mask, [[cutout_length + start[0], 32 - start[0]],
                             [cutout_length + start[1], 32 - start[1]]])
        mask = mask[cutout_length:cutout_length + 32, cutout_length:
                    cutout_length + 32]
        mask = tf.reshape(mask, [32, 32, 1])
        mask = tf.tile(mask, [1, 1, 3])
        x = tf.where(tf.equal(mask, 0), x=x, y=tf.zeros_like(x))
    return x, label


def _read_data(data_path, train_files):
    """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
    images, labels = [], []
    for file_name in train_files:
        print(file_name)
        full_name = os.path.join(data_path, file_name)
        with open(full_name, 'rb') as finp:
            data = pickle.load(finp, encoding='iso-8859-1')
            batch_images = data["data"].astype(np.float32) / 255.0
            batch_labels = np.array(data["labels"], dtype=np.int32)
            images.append(batch_images)
            labels.append(batch_labels)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    images = np.reshape(images, [-1, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])

    return images, labels


def train_reader(data_path, train_portion):

    images, labels = {}, {}

    train_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    test_file = ["test_batch", ]

    images["train"], labels["train"] = _read_data(data_path, train_files)

    num_train = len(images["train"])
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    images["valid"] = images["train"][split:num_train]
    labels["valid"] = labels["train"][split:num_train]

    images["train"] = images["train"][:split]
    labels["train"] = labels["train"][:split]

    images["test"], labels["test"] = _read_data(data_path, test_file)

    print("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std

    images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std

    return images, labels
