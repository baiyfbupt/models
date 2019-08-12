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
import numpy as np
from model_search import model


def compute_unrolled_step(image_train, label_train, image_val, label_val,
                          data_prog, startup_prog, lr, args):
    # construct model graph

    fetch = []
    unrolled_optim_prog = data_prog.clone()
    with fluid.program_guard(unrolled_optim_prog, startup_prog):
        train_logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        print(time.asctime(time.localtime(time.time())), "model define done")
        # construct unrolled model graph
        logits, unrolled_train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="unrolled_model")
        print(time.asctime(time.localtime(time.time())),
              "unrolled model define done")

        all_params = unrolled_optim_prog.global_block().all_parameters()
        model_var = utility.get_parameters(all_params, 'model')[1]
        unrolled_model_var = utility.get_parameters(all_params,
                                                    'unrolled_model')[1]

        # copy model_var to unrolled_model_var
        for m_var, um_var in zip(model_var, unrolled_model_var):
            fluid.layers.assign(m_var, um_var)

        # optimize unrolled_model_var one step (eq.9)
        unrolled_optimizer = fluid.optimizer.MomentumOptimizer(
            lr,
            args.momentum,
            regularization=fluid.regularizer.L2DecayRegularizer(
                args.weight_decay))
        unrolled_optimizer.minimize(
            unrolled_train_loss, parameter_list=unrolled_model_var)
        fetch.append(unrolled_train_loss)

    eps_prog = data_prog.clone()
    with fluid.program_guard(eps_prog, startup_prog):
        train_logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        logits, unrolled_valid_loss = model(
            image_val,
            label_val,
            args.init_channels,
            args.class_num,
            args.layers,
            name="unrolled_model")

        model_var = utility.get_parameters(
            eps_prog.global_block().all_parameters(), 'model')[1]
        unrolled_model_var = utility.get_parameters(
            eps_prog.global_block().all_parameters(), 'unrolled_model')[1]
        # get unrolled_valid_loss grad: \nabla{w'}L_val(w', a)
        valid_grads = fluid.gradients(unrolled_valid_loss, unrolled_model_var)
        # get \epsilion(eq. 10-11): 0.01/global_norm(valid_grads)
        eps = 1e-2 * fluid.layers.rsqrt(
            fluid.layers.sums([
                fluid.layers.reduce_sum(fluid.layers.square(valid_grad))
                for valid_grad in valid_grads
            ]))
        params_grads = list(zip(model_var, valid_grads))

    model_plus_prog = eps_prog.clone()
    with fluid.program_guard(model_plus_prog, startup_prog):
        # w+ = w + eps*dw`
        eps = model_plus_prog.global_block().var(eps.name)
        optimizer_pos = fluid.optimizer.SGDOptimizer(eps)
        optimizer_pos.apply_gradients(params_grads)
        print(time.asctime(time.localtime(time.time())), "w+ apply_grad done")
        fetch.append(unrolled_valid_loss)

    pos_grad_prog = data_prog.clone()
    with fluid.program_guard(pos_grad_prog, startup_prog):
        logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        # get \grad_{a}L_train(w+, a)
        arch_var = utility.get_parameters(
            pos_grad_prog.global_block().all_parameters(), 'arch')[1]
        train_grads_pos = fluid.gradients(train_loss, arch_var)
        grads_pos = [fluid.layers.assign(v) for v in train_grads_pos]
        for v in grads_pos:
            v.persistable = True
        print(time.asctime(time.localtime(time.time())), "train_gards_pos")
        fetch.append(train_loss)

    model_minus_prog = eps_prog.clone()
    with fluid.program_guard(model_minus_prog, startup_prog):
        # w- = w - eps*dw`"""
        eps = model_minus_prog.global_block().var(eps.name)
        optimizer_neg = fluid.optimizer.SGDOptimizer(-2 * eps)
        optimizer_neg.apply_gradients(params_grads)
        print(time.asctime(time.localtime(time.time())), "w- apply_grad done")
        fetch.append(unrolled_valid_loss)

    neg_grad_prog = data_prog.clone()
    with fluid.program_guard(neg_grad_prog, startup_prog):
        logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        # get \grad_{a}L_train(w-, a)
        arch_var = utility.get_parameters(
            neg_grad_prog.global_block().all_parameters(), 'arch')[1]
        train_grads_neg = fluid.gradients(train_loss, arch_var)
        grads_neg = [fluid.layers.assign(v) for v in train_grads_neg]
        for v in grads_neg:
            v.persistable = True
        print(time.asctime(time.localtime(time.time())), "train_gards_neg")
        fetch.append(train_loss)

    arch_optim_prog = eps_prog.clone()
    with fluid.program_guard(arch_optim_prog, startup_prog):
        logits, unrolled_valid_loss = model(
            image_val,
            label_val,
            args.init_channels,
            args.class_num,
            args.layers,
            name="unrolled_model")
        # recover w
        #unrolled_valid_loss = arch_optim_prog.global_block().var(unrolled_valid_loss.name)
        eps = arch_optim_prog.global_block().var(eps.name)
        optimizer_back = fluid.optimizer.SGDOptimizer(eps)
        optimizer_back.apply_gradients(params_grads)
        print(time.asctime(time.localtime(time.time())), "w apply_grad done")
        arch_var = utility.get_parameters(
            arch_optim_prog.global_block().all_parameters(), 'arch')[1]
        leader_opt = fluid.optimizer.Adam(
            args.arch_learning_rate,
            0.5,
            0.999,
            regularization=fluid.regularizer.L2DecayRegularizer(
                args.arch_weight_decay))
        leader_grads = leader_opt.backward(
            unrolled_valid_loss, parameter_list=[v.name for v in arch_var])
        print(time.asctime(time.localtime(time.time())), "leader grad done")

        arch_grads_pos = [
            arch_optim_prog.global_block()._clone_variable(
                pos_grad_prog.global_block().var(v.name),
                force_persistable=True) for v in grads_pos
        ]
        arch_grads_neg = [
            arch_optim_prog.global_block()._clone_variable(
                neg_grad_prog.global_block().var(v.name),
                force_persistable=True) for v in grads_neg
        ]

        # get final a'grad(eq. 13)
        for i, (var, grad) in enumerate(leader_grads):
            leader_grads[i] = (var, grad - (
                (arch_grads_pos[i] - arch_grads_neg[i]) / (2 * eps)) * lr)
        leader_opt.apply_gradients(leader_grads)
        print(time.asctime(time.localtime(time.time())),
              "leader apply_grad done")
        fetch.append(unrolled_valid_loss)
    return unrolled_optim_prog, model_plus_prog, pos_grad_prog, model_minus_prog, neg_grad_prog, arch_optim_prog, fetch
