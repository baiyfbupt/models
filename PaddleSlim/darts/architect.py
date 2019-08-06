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
                          config_prog, umodel_prog, modelp_prog, startup_prog,
                          lr, args):
    fetch = []
    print("umodel_prog begin", time.time())
    with fluid.program_guard(umodel_prog, startup_prog):
        with fluid.unique_name.guard():
            # construct model graph
            train_logits, train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")
            print("model define done", time.time())
            # construct unrolled model graph
            logits, unrolled_train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="unrolled_model")
            print("unrolled model define done", time.time())

            all_params = umodel_prog.global_block().all_parameters()
            model_var = utility.get_parameters(all_params, 'model')[1]
            unrolled_model_var = utility.get_parameters(all_params,
                                                        'unrolled_model')[1]

            # copy model_var to unrolled_model_var
            for m_var, um_var in zip(model_var, unrolled_model_var):
                fluid.layers.assign(m_var, um_var)

            # optimize unrolled_model_var one step (eq.9)
            unrolled_optimizer = fluid.optimizer.SGDOptimizer(lr)
            unrolled_optimizer.minimize(
                unrolled_train_loss, parameter_list=unrolled_model_var)
            print("unrolled model minimize done", time.time())
            fetch.append(unrolled_train_loss)
    print("umodel_prog done", time.time())
    print("=" * 50)

    print("modelp_prog begin", time.time())
    with fluid.program_guard(modelp_prog, startup_prog):
        with fluid.unique_name.guard():
            logits, train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")
            print("model define done", time.time())
            # get updated unrolled_valid_loss: L_val(w', a)
            logits, unrolled_valid_loss = model(
                image_val,
                label_val,
                args.init_channels,
                args.class_num,
                args.layers,
                name="unrolled_model")
            print("unrolled model define done", time.time())

            all_params = modelp_prog.global_block().all_parameters()
            model_var = utility.get_parameters(all_params, 'model')[1]
            unrolled_model_var = utility.get_parameters(all_params,
                                                        'unrolled_model')[1]
            # get unrolled_valid_loss grad: \nabla{w'}L_val(w', a)
            valid_grads = fluid.gradients(unrolled_valid_loss,
                                          unrolled_model_var)
            print("unrolled model grad done", time.time())

            # get \epsilion(eq. 10-11): 0.01/global_norm(valid_grads)
            squared_valid_grads = [
                fluid.layers.reduce_sum(fluid.layers.square(valid_grad))
                for valid_grad in valid_grads
            ]
            eps = 1e-2 * fluid.layers.rsqrt(
                fluid.layers.sums(squared_valid_grads))

            params_grads = list(zip(model_var, valid_grads))
            modelm_prog = modelp_prog.clone()

            # w+ = w + eps*dw`
            # get \nabla{a}L_train(w+, a)
            optimizer_pos = fluid.optimizer.SGDOptimizer(eps)
            optimizer_pos.apply_gradients(params_grads)
            print("unrolled model apply_grad done", time.time())
            fetch.append(unrolled_valid_loss)
    print("modelp_prog done", time.time())
    print("=" * 50)
    print("modelm_prog begin", time.time())
    with fluid.program_guard(modelm_prog, startup_prog):
        with fluid.unique_name.guard():
            logits, train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")
            print("model define done", time.time())

            arch_var = utility.get_parameters(
                modelm_prog.global_block().all_parameters(), 'arch')[1]

            train_grads_pos = fluid.gradients(train_loss, arch_var)
            print("model grad done", time.time())
            model_prog = modelm_prog.clone()

            # w- = w - eps*dw`"""
            # get \nabla{a}L_train(w-, a)
            optimizer_neg = fluid.optimizer.SGDOptimizer(-2 * eps)
            optimizer_neg.apply_gradients(params_grads)
            print("model apply_grad done", time.time())
            fetch.append(train_loss)
    print("modelm_prog done", time.time())
    print("=" * 50)

    print("model_prog begin", time.time())
    with fluid.program_guard(model_prog, startup_prog):
        with fluid.unique_name.guard():
            logits, train_loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")
            print("model define done", time.time())

            arch_var = utility.get_parameters(
                model_prog.global_block().all_parameters(), 'arch')[1]

            train_grads_neg = fluid.gradients(train_loss, arch_var)
            print("model grad done", time.time())
            arch_prog = model_prog.clone()
            # recover w
            optimizer_back = fluid.optimizer.SGDOptimizer(eps)
            optimizer_back.apply_gradients(params_grads)
            print("model apply_grad done", time.time())
            fetch.append(train_loss)
    print("model_prog done", time.time())
    print("=" * 50)

    print("arch_prog begin", time.time())
    with fluid.program_guard(arch_prog, startup_prog):
        with fluid.unique_name.guard():
            leader_opt = fluid.optimizer.Adam(args.arch_learning_rate, 0.5,
                                              0.999)
            leader_grads = leader_opt.backward(
                unrolled_valid_loss, parameter_list=arch_var)
            print("leader grad done", time.time())

            # get final a'grad(eq. 13)
            for i, (var, grad) in enumerate(leader_grads):
                leader_grads[i] = (var,
                                   (grad - lr * fluid.layers.elementwise_div(
                                       train_pos[i] - train_neg[i], 2 * eps)))
            out_prog = arch_prog.clone()
            leader_opt.apply_gradients(leader_grads)
            print("leader apply_grad done", time.time())

            fetch.append(unrolled_valid_loss)
    print("arch_prog done", time.time())
    print("=" * 50)

    return modelm_prog, model_prog, arch_prog, out_prog, fetch
