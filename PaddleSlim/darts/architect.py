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
from model_search import model


def compute_unrolled_step(image_train, label_train, image_val, label_val,
                          all_params, model_var, train_loss, lr, args):
    arch_var = utility.get_parameters(all_params, 'arch')[1]
    logits, unrolled_train_loss = model(
        image_train,
        label_train,
        args.init_channels,
        args.class_num,
        args.layers,
        name="unrolled_model")
    unrolled_model_var = utility.get_parameters(all_params, 'unrolled_model')[1]
    fluid.layers.assign(model_var, unrolled_model_var)

    unrolled_optimizer = fluid.optimizer.SGDOptimizer(lr)
    unrolled_optimizer.minimize(
        unrolled_train_loss, parameter_list=unrolled_model_var)

    logits, unrolled_valid_loss = model(
        image_val,
        label_val,
        args.init_channels,
        args.class_num,
        args.layers,
        name="unrolled_model")

    valid_grads = fluid.gradients(unrolled_valid_loss, unrolled_model_var)

    r = 1e-2
    R = r * fluid.layers.rsqrt(valid_grads)

    # w+ = w + eps*dw`
    optimizer_pos = fluid.optimizer.SGDOptimizer(R)
    optimizer_pos.apply_gradients([model_var, valid_grads])
    train_grads_pos = fluid.gradients(train_loss, arch_var)

    # w- = w - eps*dw`
    optimizer_neg = fluid.optimizer.SGDOptimizer(-2 * R)
    optimizer_neg.apply_gradients([model_var, valid_grads])
    train_grad_neg = fluid.gradients(train_loss,
                                     arch_var)  # is train_loss updated?

    # recover w
    optimizer_back = fluid.optimizer.SGDOptimizer(R)
    optimizer_back.apply_gradients([model_var, valid_grads])
