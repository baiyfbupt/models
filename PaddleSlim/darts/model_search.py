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
import sys

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *


def mixed_op(x, c_out, stride, index, reduction):
    param_attr = ParamAttr(
        name="arch_var/weight{}_{}".format(2 if reduction else 1, index))
    weight = fluid.layers.create_parameter(
        shape=[len(PRIMITIVES)],
        dtype="float32",
        attr=param_attr,
        default_initializer=Normal(
            loc=0, scale=1e-6))
    weight = fluid.layers.softmax(weight)
    ops = []
    index = 0
    for primitive in PRIMITIVES:
        op = OPS[primitive](x, c_out, stride, False)
        if 'pool' in primitive:
            op = fluid.layers.batch_norm(op)
        w = fluid.layers.slice(
            weight, axis=[0], starts=[index], ends=[index + 1])
        ops.append(fluid.layers.elementwise_mul(op, w))
        index += 1
    return fluid.layers.sums(ops)


def cell(s0, s1, steps, multiplier, c_out, reduction, reduction_prev):
    if reduction_prev:
        s0 = factorized_reduce(s0, c_out, affine=False)
    else:
        s0 = relu_conv_bn(s0, c_out, 1, 1, 0, affine=False)
    s1 = relu_conv_bn(s1, c_out, 1, 1, 0, affine=False)
    state = [s0, s1]
    offset = 0
    for i in range(steps):
        temp = []
        for j in range(2 + i):
            stride = 2 if reduction and j < 2 else 1
            temp.append(
                mixed_op(stride[j], c_out, stride, offset + j, reduction))
        offset += len(state)
        state.append(fluid.layers.sums(temp))
    out = fluid.layers.concat(input=state[-multiplier:], axis=1)
    return out


def model(x,
          y,
          is_training,
          c_in,
          num_classes,
          layers,
          steps=4,
          multiplier=4,
          stem_multiplier=3,
          name="model"):
    c_curr = stem_multiplier * c_in
    s0 = fluid.layers.conv2d(x, c_curr, 3, padding=1, bias_attr=False)
    s0 = fluid.layers.batch_norm(s0)
    s1 = fluid.layers.conv2d(x, c_curr, 3, padding=1, bias_attr=False)
    s1 = fluid.layers.batch_norm(s1)
    reduction_prev = False
    for i in range(layers):
        if i in [layers // 3, 2 * layers // 3]:
            c_curr *= 2
            reduction = True
        else:
            reduction = False
        s0, s1 = s1, cell(s0, s1, steps, multiplier, c_curr, reduction,
                          reduction_prev)
        reduction_prev = reduction
    out = fluid.layers.pool2d(s1, pool_type='avg', global_pooling=True)
    out = fluid.layers.squeeze(out, [2, 3])
    logits = fluid.layers.fc(out, num_classes)
    train_loss = fluid.layers.reduce_mean(
        fluid.layers.softmax_with_cross_entropy(logits, y))
    return logits, train_loss
