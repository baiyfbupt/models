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

import os
import sys
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

INT_MAX = sys.maxsize >> 32

OPS = {
    'none':
    lambda x, C, stride, affine, name: zero(x, stride),
    'avg_pool_3x3':
    lambda x, C, stride, affine, name: fluid.layers.pool2d(
        input=x,
        pool_size=[3, 3],
        pool_type="avg",
        pool_stride=stride,
        pool_padding=1),
    'max_pool_3x3':
    lambda x, C, stride, affine, name: fluid.layers.pool2d(
        input=x,
        pool_size=[3, 3],
        pool_type="max",
        pool_stride=stride,
        pool_padding=1),
    'skip_connect':
    lambda x, C, stride, affine, name: identity(x)
    if stride == [1, 1] else factorized_reduce(x, C, affine, name),
    'sep_conv_3x3':
    lambda x, C, stride, affine, name: sep_conv(x, C, [3, 3], stride, 1,
                                                       affine, name),
    'sep_conv_5x5':
    lambda x, C, stride, affine, name: sep_conv(x, C, [5, 5], stride, 2,
                                                       affine, name),
    'sep_conv_7x7':
    lambda x, C, stride, affine, name: sep_conv(x, C, [7, 7], stride, 3,
                                                       affine, name),
    'dil_conv_3x3':
    lambda x, C, stride, affine, name: dil_conv(x, C, [3, 3], stride, 2,
                                                       2, affine, name),
    'dil_conv_5x5':
    lambda x, C, stride, affine, name: dil_conv(x, C, [5, 5], stride, 4,
                                                       2, affine, name),
    'conv_7x1_1x7':
    lambda x, C, stride, affine, name: conv_7x1_1x7(
        x, C, stride, affine, name),
}


def bn_param_config(name=None, affine=False, op=None):
    gama_name = name + "/" + str(op) + "/gama"
    beta_name = name + "/" + str(op) + "/beta"
    if affine is True:
        gama = ParamAttr(name=gama_name)
        beta = ParamAttr(name=beta_name)
    else:
        gama = ParamAttr(
            name=gama_name,
            initializer=fluid.initializer.Constant(value=1),
            trainable=False)
        beta = ParamAttr(
            name=beta_name,
            initializer=fluid.initializer.Constant(value=0),
            trainable=False)
    return gama, beta


def zero(x, stride):
    pooled = fluid.layers.pool2d(input=x, pool_size=1, pool_stride=[2, 2])
    x = fluid.layers.zeros_like(x) if stride == [
        1, 1
    ] else fluid.layers.zeros_like(pooled)
    return x


def identity(x):
    return x


def factorized_reduce(x, c_out, affine=True, name=None):
    assert c_out % 2 == 0
    x = fluid.layers.relu(x)
    x_sliced = fluid.layers.slice(x, [2, 3], [1, 1], [INT_MAX, INT_MAX])
    conv1 = fluid.layers.conv2d(
        x,
        c_out // 2,
        1,
        stride=2,
        param_attr=fluid.ParamAttr(name=name + "/fr_conv1"),
        bias_attr=False)
    conv2 = fluid.layers.conv2d(
        x_sliced,
        c_out // 2,
        1,
        stride=2,
        param_attr=fluid.ParamAttr(name=name + "/fr_conv2"),
        bias_attr=False)
    x = fluid.layers.concat(input=[conv1, conv2], axis=1)
    gama, beta = bn_param_config(name, affine, "fr_bn")
    x = fluid.layers.batch_norm(x, param_attr=gama, bias_attr=beta)
    return x


def sep_conv(x, c_out, kernel_size, stride, padding, affine=True, name=None):
    c_in = x.shape[1]
    x = fluid.layers.relu(x)
    x = fluid.layers.conv2d(
        x,
        c_in,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=c_in,
        param_attr=fluid.ParamAttr(name=name + "/" + "sep_conv_1_1"),
        bias_attr=False)
    x = fluid.layers.conv2d(
        x,
        c_in,
        1,
        padding=0,
        param_attr=fluid.ParamAttr(name=name + "/" + "fr_conv_1_2"),
        bias_attr=False)
    gama, beta = bn_param_config(name, affine, "sep_conv_bn1")
    x = fluid.layers.batch_norm(x, param_attr=gama, bias_attr=beta)
    x = fluid.layers.relu(x)
    x = fluid.layers.conv2d(
        x,
        c_in,
        kernel_size,
        stride=1,
        padding=padding,
        groups=c_in,
        param_attr=fluid.ParamAttr(name=name + "/" + "fr_conv2_1"),
        bias_attr=False)
    x = fluid.layers.conv2d(
        x,
        c_in,
        1,
        padding=0,
        param_attr=fluid.ParamAttr(name=name + "/" + "fr_conv2_2"),
        bias_attr=False)
    gama, beta = bn_param_config(name, affine, "sep_conv_bn2")
    x = fluid.layers.batch_norm(x, param_attr=gama, bias_attr=beta)
    return x


def dil_conv(x,
             c_out,
             kernel_size,
             stride,
             padding,
             dilation,
             affine=True,
             name=None):
    c_in = x.shape[1]
    x = fluid.layers.relu(x)
    x = fluid.layers.conv2d(
        x,
        c_in,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=c_in,
        param_attr=fluid.ParamAttr(name=name + "/" + "dil_conv1"),
        bias_attr=False)
    x = fluid.layers.conv2d(
        x,
        c_out,
        1,
        padding=0,
        param_attr=fluid.ParamAttr(name=name + "/" + "dil_conv2"),
        bias_attr=False)
    gama, beta = bn_param_config(name, affine, "dil_conv_bn")
    x = fluid.layers.batch_norm(x, param_attr=gama, bias_attr=beta)
    return x


def conv_7x1_1x7(x, c_out, stride, affine=True, name=None):
    x = fluid.layers.relu(x)
    x = fluid.layers.conv2d(
        x,
        c_out, (1, 7),
        padding=(0, 3),
        param_attr=fluid.ParamAttr(name=name + "/" + "conv_7x1_1x7_1"),
        bias_attr=False)
    x = fluid.layers.conv2d(
        x,
        c_out, (7, 1),
        padding=(3, 0),
        param_attr=fluid.ParamAttr(name=name + "/" + "conv_7x1_1x7_2"),
        bias_attr=False)
    gama, beta = bn_param_config(name, affine, "conv_7x1_1x7_bn")
    x = fluid.layers.batch_norm(x, param_attr=gama, bias_attr=beta)
    return x


def relu_conv_bn(x, c_out, kernel_size, stride, padding, affine=True,
                 name=None):
    x = fluid.layers.relu(x)
    x = fluid.layers.conv2d(
        x,
        c_out,
        kernel_size,
        stride=stride,
        padding=padding,
        param_attr=fluid.ParamAttr(name=name + "/" + "rcb_conv"),
        bias_attr=False)
    gama, beta = bn_param_config(name, affine, "rcb_bn")
    x = fluid.layers.batch_norm(x, param_attr=gama, bias_attr=beta)
    return x
