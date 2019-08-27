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
import time
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import UniformInitializer, ConstantInitializer
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *


def conv_bn(x, c_out, kernel_size, padding, stride, name):
    k = (1. / x.shape[1] / kernel_size / kernel_size)**0.5
    conv1 = fluid.layers.conv2d(
        x,
        c_out,
        kernel_size,
        stride=stride,
        padding=padding,
        param_attr=fluid.ParamAttr(
            name=name + "/conv", initializer=UniformInitializer(
                low=-k, high=k)),
        bias_attr=False)
    bn1 = fluid.layers.batch_norm(
        conv1,
        param_attr=fluid.ParamAttr(
            name=name + "/bn_scale", initializer=ConstantInitializer(value=1)),
        bias_attr=fluid.ParamAttr(
            name=name + "/bn_offset", initializer=ConstantInitializer(value=0)),
        moving_mean_name=name + "/bn_mean",
        moving_variance_name=name + "/bn_variance")
    return bn1


def classifier(x, num_classes, name):
    out = fluid.layers.pool2d(x, pool_type='avg', global_pooling=True)
    out = fluid.layers.squeeze(out, axes=[2, 3])
    k = (1. / out.shape[1])**0.5
    out = fluid.layers.fc(out,
                          num_classes,
                          param_attr=fluid.ParamAttr(
                              name=name + "/fc_weights",
                              initializer=UniformInitializer(
                                  low=-k, high=k)),
                          bias_attr=fluid.ParamAttr(
                              name=name + "/fc_bias",
                              initializer=UniformInitializer(
                                  low=-k, high=k)))
    return out


def drop_path(x, drop_prob, args):
    if drop_prob > 0:
        keep_prob = 1 - drop_prob
        mask = fluid.layers.assign(
            np.random.binomial(
                1, keep_prob, size=args.batch_size).astype(np.float32))
        x = fluid.layers.elementwise_mul(x / keep_prob, mask, axis=0)
    return x


def cell(s0, s1, is_train, genotype, c_curr, reduction, reduction_prev,
         drop_prob, args, name):
    if reduction:
        op_names, indices = zip(*genotype.reduce)
        concat = genotype.reduce_concat
    else:
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
    num_cells = len(op_names) // 2
    multiplier = len(concat)

    if reduction_prev:
        s0 = factorized_reduce(s0, c_curr, name=name + '/s-2')
    else:
        s0 = relu_conv_bn(s0, c_curr, 1, 1, 0, name=name + '/s-2')
    s1 = relu_conv_bn(s1, c_curr, 1, 1, 0, name=name + '/s-1')

    state = [s0, s1]
    for i in range(num_cells):
        stride = 2 if reduction and indices[2 * i] < 2 else 1
        h1 = OPS[op_names[2 * i]](state[indices[2 * i]], c_curr, stride, True,
                                  name + "/s" + str(i) + "/h1")
        stride = 2 if reduction and indices[2 * i + 1] < 2 else 1
        h2 = OPS[op_names[2 * i + 1]](state[indices[2 * i + 1]], c_curr, stride,
                                      True, name + "/s" + str(i) + "/h2")
        h1 = fluid.layers.dropout(h1, args.dropout)
        h2 = fluid.layers.dropout(h2, args.dropout)
        if is_train and drop_prob > 0:
            if op_names[2 * i] is not 'skip_connect':
                h1 = drop_path(h1, drop_prob, args)
            if op_names[2 * i + 1] is not 'skip_connect':
                h2 = drop_path(h2, drop_prob, args)
        state.append(h1 + h2)
    out = fluid.layers.concat(input=state[-multiplier:], axis=1)
    return out


def auxiliary_cifar(x, num_classes, name):
    x = fluid.layers.relu(x)
    pooled = fluid.layers.pool2d(
        x, pool_size=5, pool_stride=3, pool_padding=0, pool_type='avg')
    conv1 = conv_bn(
        x=pooled,
        c_out=128,
        kernel_size=1,
        padding=0,
        stride=1,
        name=name + '/conv_bn1')
    # k = (1. / pooled.shape[1] / 1 / 1)**0.5
    # conv1 = fluid.layers.conv2d(
    #     pooled,
    #     128,
    #     1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_1",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # conv1 = fluid.layers.batch_norm(
    #     conv1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn1_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn1_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn1_mean",
    #     moving_variance_name=name + "/bn1_variance")
    conv1 = fluid.layers.relu(conv1)
    conv2 = conv_bn(
        x=conv1,
        c_out=768,
        kernel_size=2,
        padding=0,
        stride=1,
        name=name + '/conv_bn2')
    # k = (1. / conv1.shape[1] / 2 / 2)**0.5
    # conv2 = fluid.layers.conv2d(
    #     conv1,
    #     768,
    #     2,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_2",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # conv2 = fluid.layers.batch_norm(
    #     conv2,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn2_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn2_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn2_mean",
    #     moving_variance_name=name + "/bn2_variance")
    conv2 = fluid.layers.relu(conv2)
    out = classifier(conv2, num_classes, name)
    # out = fluid.layers.pool2d(conv2, pool_type='avg', global_pooling=True)
    # out = fluid.layers.squeeze(out, axes=[2, 3])
    # k = (1. / out.shape[1])**0.5
    # out = fluid.layers.fc(out,
    #                       num_classes,
    #                       param_attr=fluid.ParamAttr(
    #                           name=name + "/fc_weights",
    #                           initializer=UniformInitializer(
    #                               low=-k, high=k)),
    #                       bias_attr=fluid.ParamAttr(
    #                           name=name + "/fc_bias",
    #                           initializer=UniformInitializer(
    #                               low=-k, high=k)))
    return out


def network_cifar(x, is_train, c_in, num_classes, layers, auxiliary, genotype,
                  stem_multiplier, drop_prob, args, name):
    c_curr = stem_multiplier * c_in
    x = conv_bn(
        x=x,
        c_out=c_curr,
        kernel_size=3,
        padding=1,
        stride=1,
        name=name + '/s0/conv_bn')
    # k = (1. / x.shape[1] / 3 / 3)**0.5
    # x = fluid.layers.conv2d(
    #     x,
    #     c_curr,
    #     3,
    #     padding=1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_0",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # x = fluid.layers.batch_norm(
    #     x,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn0_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn0_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn0_mean",
    #     moving_variance_name=name + "/bn0_variance")
    s0 = s1 = x
    reduction_prev = False
    logits_aux = None
    c_curr = c_in
    for i in range(layers):
        if i in [layers // 3, 2 * layers // 3]:
            c_curr *= 2
            reduction = True
        else:
            reduction = False
        s0, s1 = s1, cell(s0, s1, is_train, genotype, c_curr, reduction,
                          reduction_prev, drop_prob, args, name + "/l" + str(i))
        reduction_prev = reduction
        if i == 2 * layers // 3:
            if auxiliary and is_train:
                logits_aux = auxiliary_cifar(s1, num_classes,
                                             name + "/l" + str(i) + "/aux")

    logits = classifier(s1, num_classes, name)
    # out = fluid.layers.pool2d(s1, pool_type='avg', global_pooling=True)
    # out = fluid.layers.squeeze(out, axes=[2, 3])
    # k = (1. / out.shape[1])**0.5
    # logits = fluid.layers.fc(out,
    #                          num_classes,
    #                          param_attr=fluid.ParamAttr(
    #                              name=name + "/fc_weights",
    #                              initializer=UniformInitializer(
    #                                  low=-k, high=k)),
    #                          bias_attr=fluid.ParamAttr(
    #                              name=name + "/fc_bias",
    #                              initializer=UniformInitializer(
    #                                  low=-k, high=k)))
    return logits, logits_aux


def auxiliary_imagenet(x, num_classes, name):
    x = fluid.layers.relu(x)
    pooled = fluid.layers.pool2d(
        x, pool_size=5, pool_stride=2, pool_padding=0, pool_type='avg')
    conv1 = conv_bn(
        x=pooled,
        c_out=128,
        kernel_size=1,
        padding=0,
        stride=1,
        name=name + '/conv_bn1')
    # k = (1. / pooled.shape[1] / 1 / 1)**0.5
    # conv1 = fluid.layers.conv2d(
    #     pooled,
    #     128,
    #     1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_1",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # conv1 = fluid.layers.batch_norm(
    #     conv1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn1_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn1_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn1_mean",
    #     moving_variance_name=name + "/bn1_variance")
    conv1 = fluid.layers.relu(conv1)
    conv2 = conv_bn(
        x=conv1,
        c_out=768,
        kernel_size=2,
        padding=0,
        stride=1,
        name=name + '/conv_bn2')
    # k = (1. / conv1.shape[1] / 2 / 2)**0.5
    # conv2 = fluid.layers.conv2d(
    #     conv1,
    #     768,
    #     2,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_2",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # conv2 = fluid.layers.batch_norm(
    #     conv2,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn2_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn2_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn2_mean",
    #     moving_variance_name=name + "/bn2_variance")
    conv2 = fluid.layers.relu(conv2)
    out = classifier(conv2, num_classes, name)

    # out = fluid.layers.pool2d(conv2, pool_type='avg', global_pooling=True)
    # out = fluid.layers.squeeze(out, axes=[2, 3])
    # k = (1. / out.shape[1])**0.5
    # out = fluid.layers.fc(out,
    #                       num_classes,
    #                       param_attr=fluid.ParamAttr(
    #                           name=name + "/fc_weights",
    #                           initializer=UniformInitializer(
    #                               low=-k, high=k)),
    #                       bias_attr=fluid.ParamAttr(
    #                           name=name + "/fc_bias",
    #                           initializer=UniformInitializer(
    #                               low=-k, high=k)))
    return out


def network_imagenet(x, is_train, c_in, num_classes, layers, auxiliary,
                     genotype, stem_multiplier, drop_prob, args, name):
    x = conv_bn(
        x=x,
        c_out=c_in // 2,
        kernel_size=3,
        padding=1,
        stride=2,
        name=name + '/conv_bn/s0_0')
    # k = (1. / x.shape[1] / 3 / 3)**0.5
    # x = fluid.layers.conv2d(
    #     x,
    #     c_in // 2,
    #     3,
    #     padding=1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_0_0",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # x = fluid.layers.batch_norm(
    #     x,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn0_0_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn0_0_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn0_0_mean",
    #     moving_variance_name=name + "/bn0_0_variance")
    x = fluid.layers.relu(x)
    s0 = conv_bn(
        x=x,
        c_out=c_in,
        kernel_size=3,
        padding=1,
        stride=2,
        name=name + '/conv_bn/s0_1')
    # x = fluid.layers.conv2d(
    #     x,
    #     c_in,
    #     3,
    #     padding=1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_0_1",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # s0 = fluid.layers.batch_norm(
    #     x,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn0_1_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn0_1_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn0_1_mean",
    #     moving_variance_name=name + "/bn0_1_variance")
    s0 = fluid.layers.relu(s0)
    s1 = conv_bn(
        x=s0,
        c_out=c_in,
        kernel_size=3,
        padding=1,
        stride=2,
        name=name + '/conv_bn/s1')
    # s0 = fluid.layers.conv2d(
    #     s0,
    #     c_in,
    #     3,
    #     padding=1,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/conv_1",
    #         initializer=UniformInitializer(
    #             low=-k, high=k)),
    #     bias_attr=False)
    # s1 = fluid.layers.batch_norm(
    #     s0,
    #     param_attr=fluid.ParamAttr(
    #         name=name + "/bn1_scale", initializer=ConstantInitializer(value=1)),
    #     bias_attr=fluid.ParamAttr(
    #         name=name + "/bn1_offset",
    #         initializer=ConstantInitializer(value=0)),
    #     moving_mean_name=name + "/bn1_mean",
    #     moving_variance_name=name + "/bn1_variance")
    reduction_prev = False
    logits_aux = None
    c_curr = c_in
    for i in range(layers):
        if i in [layers // 3, 2 * layers // 3]:
            c_curr *= 2
            reduction = True
        else:
            reduction = False
        s0, s1 = s1, cell(s0, s1, is_train, genotype, c_curr, reduction,
                          reduction_prev, drop_prob, args, name + "/l" + str(i))
        reduction_prev = reduction
        if i == 2 * layers // 3:
            if auxiliary and is_train:
                logits_aux = auxiliary_imagenet(s1, num_classes,
                                                name + "/l" + str(i) + "/aux")

    logits = classifier(s1, num_classes, name)
    # out = fluid.layers.pool2d(s1, pool_type='avg', global_pooling=True)
    # out = fluid.layers.squeeze(out, axes=[2, 3])
    # k = (1. / out.shape[1])**0.5
    # logits = fluid.layers.fc(out,
    #                          num_classes,
    #                          param_attr=fluid.ParamAttr(
    #                              name=name + "/fc_weights",
    #                              initializer=UniformInitializer(
    #                                  low=-k, high=k)),
    #                          bias_attr=fluid.ParamAttr(
    #                              name=name + "/fc_bias",
    #                              initializer=UniformInitializer(
    #                                  low=-k, high=k)))
    return logits, logits_aux
