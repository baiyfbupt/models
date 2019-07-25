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
add_arg('data',              str,   '../data/cifar-10-batches-py', "The dir of dataset.")
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
add_arg('image_shape',       str,   "3,224,224",     "input image size")
add_arg('with_mem_opt',      bool,  True,            "Whether to use memory optimization or not.")
parser.add_argument('--cutout',   action='store_true', help='If set, use cutout.')
parser.add_argument('--unrolled', action='store_true', help='If set, use one-step unrolled validation loss.')
#yapf: enable

output_dir = '/outputs/train_model/'
if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)

CIFAR10 = 50000


def build_program(main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shape=[[-1] + image_shape, [-1, 1], [-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0, 0, 0],
            dtypes=["float32", "int64", "float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image_train, label_train, image_val, label_val = fluid.layers.read_file(
                py_reader)
    return image_train, label_train, image_val, label_val, py_reader


def main(args):
    model_save_dir = args.model_save_dir
    with_memory_optimization = args.with_mem_opt

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    batch_size_per_device = args.batch_size // devices_num
    is_shuffle = True

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    image_train, label_train, image_val, label_val, py_reader = build_program(
        main_prog=train_prog, startup_prog=startup_prog, args=args)

    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    step_per_epoch = int(CIFAR10 / args.batch_size)
    learning_rate = fluid.layers.cosine_decay(args.learning_rate, step_per_epoch,
                                   args.epochs)

    all_params = train_prog.global_block().all_parameters()

    model_var = utility.get_parameters(all_params, 'model')[1]

    unrolled_valid_loss = architect.compute_unrolled_step(image_train, label_train, image_val,
                                    label_val, all_params, model_var, learning_rate, args)
    follower_opt = fluid.optimizer.MomentumOptimizer(learning_rate, args.momentum)
    train_logits, train_loss = model(image_train, label_train, args.init_channels,
                               args.class_num, args.layers)
    train_top1 = fluid.layers.accuracy(input=train_logits, label = label_train, k=1)
    train_top5 = fluid.layers.accuracy(input=train_logits, label = label_train, k=5)

    follower_grads = fluid.gradients(train_loss, model_var)
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0),
        param_list=model_var)
    follower_opt.apply_gradients([model_var, follower_grads])

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if with_memory_optimization:
        fluid.memory_optimize(train_prog)

    train_reader = reader.train_val(args, batch_size_per_device, args.train_portion, is_shuffle)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 1
    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=args.use_gpu,
        loss_name=train_loss.name,
        exec_strategy=exec_strategy)

    def save_model(postfix, program):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    def infer(epoch_id):
        infer_logits, infer_loss = model(image_val, label_val, args.init_channels,args.class_num,args.layers)
        infer_top1 = fluid.layers.accuracy(input=infer_logits, label = label_train, k=1)
        infer_top5 = fluid.layers.accuracy(input=infer_logits, label = label_train, k=5)
        test_fetch_list = [infer_loss, infer_top1, infer_top5]
        loss = utility.AvgrageMeter()
        top1 = utility.AvgrageMeter()
        top5 = utility.AvgrageMeter()
        py_reader.start()
        step_id = 0
        try:
            while True:
                loss_v, top1_v, top5 = exe.run(test_prog, fetch_list=test_fetch_list)
                loss.update(np.array(loss_v), args.batch_size)
                top1.update(np.array(top1_v), args.batch_size)
                top5.update(np.array(top5_v), args.batch_size)
                print("Epoch {}, Step {}, loss {}, acc_1 {}, acc_5 {}".format(epoch_id, step_id, loss.avg, top1.avg, top5.avg))
                step_id += 1
        except fluid.core.EOFException:
            py_reader.reset()
        print("Epoch {0}, top1 {1}, top5 {2}".format(epoch_id, top1.avg, top5.avg))


    fetch_list = [train_loss, unrolled_valid_loss, train_top1, train_top5]
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for epoch_id in range(args.epochs):
        py_reader.start()
        step_id = 0
        try:
            while True: # train
                loss_v, _, top1_v, top5_v = train_exe.run(fetch_list = [v.name for v in fetch_list])
                loss.update(np.array(loss_v), args.batch_size)
                top1.update(np.array(top1_v), args.batch_size)
                top5.update(np.array(top5_v), args.batch_size)
                print("Epoch {}, Step {}, loss {}, acc_1 {}, acc_5 {}".format(epoch_id, step_id, loss.avg, top1.avg, top5.avg))
                step_id += 1
                sys.stdout.flush()
        except (fluid.core.COFException, StopIteration):
            py_reader.reset()
            break
        infer(epoch_id)


if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)