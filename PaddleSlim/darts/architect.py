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

import paddle.fluid as fluid
import utility
import time
from model_search import model


def compute_unrolled_step(image_train, label_train, image_val, label_val,
                          train_prog, lr, args):
    print("enter compute_unrolled_step", time.time())
    train_logits, train_loss = model(image_train, label_train,
                                     args.init_channels, args.class_num,
                                     args.layers)
    print("model done", time.time())
    logits, unrolled_train_loss = model(
        image_train,
        label_train,
        args.init_channels,
        args.class_num,
        args.layers,
        name="unrolled_model")
    print("unrolled_model done", time.time())

    all_params = train_prog.global_block().all_parameters()
    model_var = utility.get_parameters(all_params, 'model')[1]
    arch_var = utility.get_parameters(all_params, 'arch')[1]
    unrolled_model_var = utility.get_parameters(all_params, 'unrolled_model')[1]
    print("get_parameters done", time.time())

    for m_var, um_var in zip(model_var, unrolled_model_var):
        fluid.layers.assign(m_var, um_var)

    print("paramters assign done", time.time())
    unrolled_optimizer = fluid.optimizer.SGDOptimizer(lr)
    unrolled_optimizer.minimize(
        unrolled_train_loss, parameter_list=unrolled_model_var)

    print("before unrolled model", time.time())
    logits, unrolled_valid_loss = model(
        image_val,
        label_val,
        args.init_channels,
        args.class_num,
        args.layers,
        name="unrolled_model")

    print("unrolled_model done", time.time())
    valid_grads = fluid.gradients(unrolled_valid_loss, unrolled_model_var)
    print("get valid_grads done", time.time())

    squared_valid_grads = [
        fluid.layers.reduce_sum(fluid.layers.square(valid_grad))
        for valid_grad in valid_grads
    ]
    eps = 1e-2 * fluid.layers.rsqrt(fluid.layers.sums(squared_valid_grads))
    print("get eps done", time.time())

    params_grads = list(zip(model_var, valid_grads))
    # w+ = w + eps*dw`
    optimizer_pos = fluid.optimizer.SGDOptimizer(eps)
    optimizer_pos.apply_gradients(params_grads)
    print("before model", time.time())
    logits, train_loss = model(image_train, label_train, args.init_channels,
                               args.class_num, args.layers)
    print("model done", time.time())
    #print("arch_var: ", arch_var)
    train_grads_pos = fluid.gradients(train_loss, arch_var)
    print("train_grads_pos done", time.time())

    # w- = w - eps*dw`
    optimizer_neg = fluid.optimizer.SGDOptimizer(-2 * eps)
    optimizer_neg.apply_gradients(params_grads)
    print("before model", time.time())
    logits, train_loss = model(image_train, label_train, args.init_channels,
                               args.class_num, args.layers)
    print("model done", time.time())
    train_grads_neg = fluid.gradients(train_loss, arch_var)

    # recover w
    optimizer_back = fluid.optimizer.SGDOptimizer(eps)
    optimizer_back.apply_gradients(params_grads)
    leader_opt = fluid.optimizer.Adam(args.arch_learning_rate, 0.5, 0.999)
    leader_grads = leader_opt.backward(
        unrolled_valid_loss, parameter_list=arch_var)

    for i, (var, grad) in enumerate(leader_grads):
        leader_grads[i] = (var, (
            grad - args.learning_rate * fluid.layers.elementwise_div(
                train_grads_pos[i] - train_grads_neg[i], 2 * R)))
    leader_opt.apply_gradients(leader_grads)

    return unrolled_valid_loss
