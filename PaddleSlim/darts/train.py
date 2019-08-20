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


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
set_paddle_flags({
    'FLAGS_eager_delete_tensor_gb': 0,  # enable GC
    # You can omit the following settings, because the default
    # value of FLAGS_memory_fraction_of_eager_deletion is 1,
    # and default value of FLAGS_fast_eager_deletion_mode is 1
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fast_eager_deletion_mode': 1,
    # Setting the default used gpu memory
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from model import network_cifar as network
import genotypes
import reader
import utility

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('profile',           bool,  False,           "Enable profiler.")
add_arg('parallel',          bool,  True,            "Whether use multi-GPU/threads.")
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('num_workers',       int,   2,               "The multiprocess reader number.")
add_arg('data',              str,   './data/cifar-10-batches-py', "The dir of dataset.")
add_arg('batch_size',        int,   96,              "Minibatch size.")
add_arg('learning_rate',     float, 0.025,           "The start learning rate.")
add_arg('learning_rate_min', float, 0.001,           "The min learning rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('weight_decay',      float, 3e-4,            "Weight_decay.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   600,             "Epoch number.")
add_arg('init_channels',     int,   36,              "Init channel number.")
add_arg('layers',            int,   20,              "Total number of layers.")
add_arg('class_num',         int,   10,              "Class number of dataset.")
add_arg('model_save_dir',    str,   'output',        "The path to save model.")
add_arg('cutout',            bool,  True,            'Whether use cutout.')
add_arg('cutout_length',     int,   16,              "Cutout length.")
add_arg('auxiliary',         bool,  True,            'Use auxiliary tower.')
add_arg('auxiliary_weight',  float, 0.4,             "Weight for auxiliary loss.")
add_arg('drop_path_prob',    float, 0.2,             "Drop path probability.")
add_arg('save',              str,   'EXP',           "Experiment name.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('train_portion',     float, 0.5,             "Portion of training data.")
add_arg('arch_learning_rate',float, 3e-4,            "Learning rate for arch encoding.")
add_arg('arch_weight_decay', float, 1e-3,            "Weight decay for arch encoding.")
add_arg('image_shape',       str,   "3,32,32",       "input image size")
add_arg('arch',              str,   'DARTS',         "which architecture to use")
add_arg('with_mem_opt',      bool,  False,           "Whether to use memory optimization or not.")
#yapf: enable

output_dir = './outputs/train_model/'
if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)
CIFAR10_TRAIN = 50000
CIFAR10_VALID = 10000


def build_program(main_prog, startup_prog, is_train, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.layers.data(name="image", shape=image_shape, dtype="float32")
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            genotype = eval("genotypes.%s" % args.arch)
            logits, logits_aux = network(x=image, is_train=is_train, c_in=args.init_channels, num_classes=args.class_num, layers=args.layers, auxiliary=args.auxiliary,genotype=genotype, stem_multiplier=3, drop_prob=0, args=args, name='model')
            top1 = fluid.layers.accuracy(input=logits, label=label, k=1)
            top5 = fluid.layers.accuracy(input=logits, label=label, k=5)
            loss = fluid.layers.reduce_mean(
                fluid.layers.softmax_with_cross_entropy(logits, label))
            if is_train:
                if args.auxiliary:
                    loss_aux = fluid.layers.reduce_mean(
                        fluid.layers.softmax_with_cross_entropy(logits_aux, label))
                    loss = loss + args.auxiliary_weight*loss_aux
                step_per_epoch = int(CIFAR10_TRAIN / args.batch_size)
                learning_rate = fluid.layers.cosine_decay(args.learning_rate, step_per_epoch, args.epochs)
                fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
                optimizer = fluid.optimizer.MomentumOptimizer(learning_rate, args.momentum, regularization=fluid.regularizer.L2DecayRegularizer(args.weight_decay))
                optimizer.minimize(loss)
                outs = [loss, top1, top5, learning_rate]
            else:
                outs = [loss, top1, top5]
    return outs

def train(main_prog, exe,  epoch_id, train_batches, fetch_list):
    step_per_epoch = int(CIFAR10_TRAIN / args.batch_size)
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for step_id in range(step_per_epoch):
        if args.profile:
            if epoch_id == 0 and step_id == 5:
                profiler.start_profiler("All")
            elif epoch_id == 0 and step_id == 10:
                profiler.stop_profiler("total", "/tmp/profile")
        image, label = next(train_batches)
        feed = {"image": image, "label": label}
        loss_v, top1_v, top5_v, lr = exe.run(main_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Train Epoch {}, Step {}, Lr {:.3f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".format(epoch_id, step_id, lr[0], loss.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0]

def valid(main_prog, exe,  epoch_id, valid_batches, fetch_list):
    step_per_epoch = int(CIFAR10_VALID / args.batch_size)
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for step_id in range(step_per_epoch):
        image, label = next(valid_batches)
        feed = {"image": image, "label": label}
        loss_v, top1_v, top5_v = exe.run(main_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Train Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0]

def main(args):
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    batch_size_per_device = args.batch_size // devices_num
    is_shuffle = True

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    train_fetch_list = build_program(main_prog=train_prog, startup_prog=startup_prog, is_train=True, args=args)
    valid_fetch_list = build_program(main_prog=test_prog, startup_prog=startup_prog, is_train=False, args=args)

    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    train_batches = reader.train_valid(batch_size_per_device, True, is_shuffle, args)()
    valid_batches = reader.train_valid(batch_size_per_device, False, False, args)()

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4
    build_strategy = fluid.BuildStrategy()
    if args.with_mem_opt:
        train_fetch_list[0].persistable = True
        train_fetch_list[1].persistable = True
        train_fetch_list[2].persistable = True
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True

    train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
                 loss_name=train_fetch_list[0].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)

    def save_model(postfix, program):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)


    for epoch_id in range(args.epochs):
        train_top1 = train(train_prog, exe, epoch_id, train_batches, train_fetch_list)
        print("Epoch {}, train_acc {:.6f}".format(epoch_id, train_top1))
        valid_top1 = valid(test_prog, exe,  epoch_id, valid_batches, valid_fetch_list)
        print("Epoch {}, valid_acc {:.6f}".format(epoch_id, valid_top1))
        # (TODO)save model


if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)
