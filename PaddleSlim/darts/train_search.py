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
import math
import time
import shutil
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
import reader
import utility
import architect
from model_search import model

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel',          bool,  True,            "Whether use multi-GPU/threads or not.")
add_arg('use_pyreader',      bool,  False,            "Whether use pyreader or not.")
add_arg('data',              str,   './data/cifar-10-batches-py', "The dir of dataset.")
add_arg('batch_size',        int,   16,              "Minibatch size.")
add_arg('learning_rate',     float, 0.025,           "The start learning rate.")
add_arg('learning_rate_min', float, 0.001,           "The min learning rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('weight_decay',      float, 3e-4,            "Weight_decay.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   50,              "Epoch number.")
add_arg('init_channels',     int,   16,              "Init channel number.")
add_arg('layers',            int,   8,               "Total number of layers.")
add_arg('class_num',         int,   10,              "Class number of dataset.")
add_arg('model_save_dir',    str,   'output',        "The path to save model.")
add_arg('cutout_length',     int,   16,              "Cutout length.")
add_arg('drop_path_prob',    float, 0.3,             "Drop path probability.")
add_arg('save',              str,   'EXP',           "Experiment name.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('train_portion',     float, 0.5,             "Portion of training data.")
add_arg('arch_learning_rate',float, 3e-4,            "Learning rate for arch encoding.")
add_arg('arch_weight_decay', float, 1e-3,            "Weight decay for arch encoding.")
add_arg('image_shape',       str,   "3,32,32",     "input image size")
add_arg('with_mem_opt',      bool,  False,            "Whether to use memory optimization or not.")
parser.add_argument('--cutout',   action='store_true', help='If set, use cutout.')
parser.add_argument('--unrolled', action='store_true', help='If set, use one-step unrolled validation loss.')
#yapf: enable

output_dir = '/outputs/train_model/'
if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)

CIFAR10 = 50000


def main(args):
    model_save_dir = args.model_save_dir

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    batch_size_per_device = args.batch_size // devices_num
    step_per_epoch = int(CIFAR10 / 2 / args.batch_size)
    is_shuffle = True

    startup_prog = fluid.Program()
    config_prog = fluid.Program()
    forward_prog = fluid.Program()

    test_prog = fluid.Program()

    image_shape = [int(m) for m in args.image_shape.split(",")]
    print("=" * 50)
    print("begin construct graph", time.time())
    with fluid.program_guard(config_prog, startup_prog):
        if args.use_pyreader == True:
            py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes=[[-1] + image_shape, [-1, 1], [-1] + image_shape, [-1, 1]],
                lod_levels=[0, 0, 0, 0],
                dtypes=["float32", "int64", "float32", "int64"],
                use_double_buffer=True)
        with fluid.unique_name.guard():
            # split data to train(for model_var optimize) and val(for arch_var optimize)
            if args.use_pyreader == True:
                image_train, label_train, image_val, label_val = fluid.layers.read_file(
                        py_reader)
            else:
                image_train = fluid.layers.data(name="image_train", shape=image_shape, dtype="float32")
                label_train = fluid.layers.data(name="label_train", shape=[1], dtype="int64")
                image_val = fluid.layers.data(name="image_val", shape=image_shape, dtype="float32")
                label_val = fluid.layers.data(name="label_val", shape=[1], dtype="int64")
            learning_rate = fluid.layers.cosine_decay(args.learning_rate, step_per_epoch,
                                    args.epochs)

    print("data define done", time.time())

    forward_prog = config_prog.clone()
    unrolled_train_prog, modelp_prog, modelm_prog, model_prog, arch_prog, fetch = architect.compute_unrolled_step(image_train, label_train, image_val,
                        label_val, forward_prog, startup_prog, args.learning_rate, args)

    print("arch optimize done", time.time())

    out_prog = config_prog.clone()
    with fluid.program_guard(out_prog, startup_prog):
        with fluid.unique_name.guard():
            train_logits, train_loss = model(image_train, label_train, args.init_channels,
                                             args.class_num, args.layers)
            train_top1 = fluid.layers.accuracy(input=train_logits, label = label_train, k=1)
            train_top5 = fluid.layers.accuracy(input=train_logits, label = label_train, k=5)

            model_var = utility.get_parameters(out_prog.global_block().all_parameters(), 'model')[1]
            # update model_var with gradientclip
            follower_opt = fluid.optimizer.MomentumOptimizer(learning_rate, args.momentum)
            #follower_opt.minimize(train_loss, parameter_list=model_var)
            follower_grads = fluid.gradients(train_loss, model_var)
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0),
                param_list=model_var)
            params_grads = list(zip(model_var, follower_grads))
            follower_opt = fluid.optimizer.MomentumOptimizer(learning_rate, args.momentum)
            follower_opt.apply_gradients(params_grads)

    print("construct graph done", time.time())
    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.use_pyreader == True:
        train_reader = reader.train_val(args, batch_size_per_device, args.train_portion, is_shuffle)
        py_reader.decorate_paddle_reader(train_reader)
    else:
        batches = reader.train_val(args, batch_size_per_device, args.train_portion, is_shuffle)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4
    build_strategy = fluid.BuildStrategy()
    if args.with_mem_opt:
        #train_loss.persistable = True
        #train_logits.persistable = True
        #train_top1.persistable = True
        #train_top5.persistable = True
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
    unrolled_train_prog = fluid.CompiledProgram(unrolled_train_prog).with_data_parallel(
                 loss_name=fetch[0].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    modelp_prog = fluid.CompiledProgram(modelp_prog).with_data_parallel(
                 loss_name=fetch[1].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    modelm_prog = fluid.CompiledProgram(modelm_prog).with_data_parallel(
                 loss_name=fetch[2].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    model_prog = fluid.CompiledProgram(model_prog).with_data_parallel(
                 loss_name=fetch[3].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    arch_prog = fluid.CompiledProgram(arch_prog).with_data_parallel(
                 loss_name=fetch[4].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    out_prog = fluid.CompiledProgram(out_prog).with_data_parallel(
                 loss_name=train_loss.name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)

    def save_model(postfix, program):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    # def infer(epoch_id):
    #     infer_logits, infer_loss = model(image_val, label_val, args.init_channels,args.class_num,args.layers)
    #     infer_top1 = fluid.layers.accuracy(input=infer_logits, label = label_train, k=1)
    #     infer_top5 = fluid.layers.accuracy(input=infer_logits, label = label_train, k=5)
    #     test_fetch_list = [infer_loss, infer_top1, infer_top5]
    #     loss = utility.AvgrageMeter()
    #     top1 = utility.AvgrageMeter()
    #     top5 = utility.AvgrageMeter()
    #     py_reader.start()
    #     step_id = 0
    #     try:
    #         while True:
    #             loss_v, top1_v, top5 = exe.run(test_prog, fetch_list=test_fetch_list)
    #             loss.update(np.array(loss_v), args.batch_size)
    #             top1.update(np.array(top1_v), args.batch_size)
    #             top5.update(np.array(top5_v), args.batch_size)
    #             print("Epoch {}, Step {}, loss {}, acc_1 {}, acc_5 {}".format(epoch_id, step_id, loss.avg, top1.avg, top5.avg))
    #             step_id += 1
    #     except fluid.core.EOFException:
    #         py_reader.reset()
    #     print("Epoch {0}, top1 {1}, top5 {2}".format(epoch_id, top1.avg, top5.avg))


    fetch_list = [learning_rate, train_loss, train_top1, train_top5]
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for epoch_id in range(args.epochs):
        if args.use_pyreader:
            py_reader.start()
            step_id = 0
            try:
                while True:
                    _ = exe.run(unrolled_train_prog, fetch_list=[fetch[0].name])
                    _ = exe.run(modelp_prog, fetch_list=[fetch[1].name])
                    _ = exe.run(modelm_prog, fetch_list=[fetch[2].name])
                    _ = exe.run(arch_prog, fetch_list=[fetch[3].name])
                    loss_v, top1_v, top5_v = exe.run(out_prog, fetch_list = [v.name for v in fetch_list])
                    loss.update(np.array(loss_v), args.batch_size)
                    top1.update(np.array(top1_v), args.batch_size)
                    top5.update(np.array(top5_v), args.batch_size)
                    print("Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[0]))
                    step_id += 1
                    sys.stdout.flush()
            except (fluid.core.EOFException):
                py_reader.reset()
                break
            # infer(epoch_id)
        else:
            for step_id in range(step_per_epoch):
                image_train, label_train, image_val, label_val = next(batches())
                #print(image_train)
                #print(label_train)
                feed = {"image_train": image_train, "label_train": label_train, "image_val": image_val, "label_val": label_val}
                _ = exe.run(unrolled_train_prog, feed=feed, fetch_list=[fetch[0].name])
                _ = exe.run(modelp_prog, feed=feed, fetch_list=[fetch[1].name])
                _ = exe.run(modelm_prog, feed=feed, fetch_list=[fetch[2].name])
                _ = exe.run(model_prog, feed=feed, fetch_list=[fetch[3].name])
                _ = exe.run(arch_prog, feed=feed, fetch_list=[fetch[4].name])
                lr_v, loss_v, top1_v, top5_v = exe.run(out_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
                loss.update(np.array(loss_v), args.batch_size)
                #loss = np.array(loss_v)
                top1.update(np.array(top1_v), args.batch_size)
                top5.update(np.array(top5_v), args.batch_size)
                lr = np.array(lr_v)
                print("Epoch {}, Step {}, Lr {:.3f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".format(epoch_id, step_id, lr[0], loss.avg[0], top1.avg[0], top5.avg[0]))
            # infer(epoch_id)



if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)
