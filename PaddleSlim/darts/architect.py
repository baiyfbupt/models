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
                          train_prog, startup_prog, lr, args):
    fetch = []
    modelp_prog = fluid.Program()
    modelm_prog = fluid.Program()
    arch_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # construct model graph
            train_logits, train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")
            # construct unrolled model graph
            logits, unrolled_train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="unrolled_model")

            all_params = train_prog.global_block().all_parameters()
            model_var = utility.get_parameters(all_params, 'model')[1]
            arch_var = utility.get_parameters(all_params, 'arch')[1]
            unrolled_model_var = utility.get_parameters(all_params,
                                                        'unrolled_model')[1]

            # copy model_var to unrolled_model_var
            for m_var, um_var in zip(model_var, unrolled_model_var):
                fluid.layers.assign(m_var, um_var)

            # optimize unrolled_model_var one step (eq.9)
            unrolled_optimizer = fluid.optimizer.SGDOptimizer(lr)
            unrolled_optimizer.minimize(
                unrolled_train_loss, parameter_list=unrolled_model_var)

            # get updated unrolled_valid_loss: L_val(w', a)
            logits, unrolled_valid_loss = model(
                image_val,
                label_val,
                args.init_channels,
                args.class_num,
                args.layers,
                name="unrolled_model")

            # get unrolled_valid_loss grad: \nabla{w'}L_val(w', a)
            valid_grads = fluid.gradients(unrolled_valid_loss,
                                          unrolled_model_var)

            # get \epsilion(eq. 10-11): 0.01/global_norm(valid_grads)
            squared_valid_grads = [
                fluid.layers.reduce_sum(fluid.layers.square(valid_grad))
                for valid_grad in valid_grads
            ]
            eps = 1e-2 * fluid.layers.rsqrt(
                fluid.layers.sums(squared_valid_grads))

            params_grads = list(zip(model_var, valid_grads))
            # w+ = w + eps*dw`
            # get \nabla{a}L_train(w+, a)
            optimizer_pos = fluid.optimizer.SGDOptimizer(eps)
            modelp_prog = train_prog.clone()
            optimizer_pos.apply_gradients(params_grads)
            fetch.append(unrolled_valid_loss)

    with fluid.program_guard(modelp_prog, startup_prog):
        with fluid.unique_name.guard():
            logits, train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")

            arch_var = utility.get_parameters(
                modelp_prog.global_block().all_parameters(), 'arch')[1]

            train_grads_pos = fluid.gradients(train_loss, arch_var)

            # w- = w - eps*dw`"""
            # get \nabla{a}L_train(w-, a)
            optimizer_neg = fluid.optimizer.SGDOptimizer(-2 * eps)
            modelm_prog = modelp_prog.clone()
            optimizer_neg.apply_gradients(params_grads)
            fetch.append(train_loss)

    with fluid.program_guard(modelm_prog, startup_prog):
        with fluid.unique_name.guard():
            logits, train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")

            arch_var = utility.get_parameters(
                modelm_prog.global_block().all_parameters(), 'arch')[1]

            train_grads_neg = fluid.gradients(train_loss, arch_var)
            # recover w
            optimizer_back = fluid.optimizer.SGDOptimizer(eps)
            arch_prog = modelm_prog.clone()
            optimizer_back.apply_gradients(params_grads)
            fetch.append(train_loss)

    with fluid.program_guard(arch_prog, startup_prog):
        with fluid.unique_name.guard():
            leader_opt = fluid.optimizer.Adam(args.arch_learning_rate, 0.5,
                                              0.999)
            leader_grads = leader_opt.backward(
                unrolled_valid_loss, parameter_list=arch_var)

            # get final a'grad(eq. 13)
            for i, (var, grad) in enumerate(leader_grads):
                leader_grads[i] = (var, (
                    grad - args.learning_rate * fluid.layers.elementwise_div(
                        train_pos[i] - train_neg[i], 2 * eps)))
            out_prog = arch_prog.clone()
            leader_opt.apply_gradients(leader_grads)

            fetch.append(unrolled_valid_loss)

    return train_prog, startup_prog, modelp_prog, modelm_prog, arch_prog, out_prog, fetch
