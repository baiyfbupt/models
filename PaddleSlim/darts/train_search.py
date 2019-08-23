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
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


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
    'FLAGS_fraction_of_gpu_memory_to_use': 0.92
})

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import reader
import utility
import architect
from model_search import model, get_genotype

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('profile',           bool,  False,           "Enable profiler.")
add_arg('report_freq',       int,   10,              "Report frequency.")
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('num_workers',       int,   8,               "The multiprocess reader number.")
add_arg('data',              str,   'cifar-10',      "The dir of dataset.")
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
add_arg('cutout',            bool,  True,            'Whether use cutout.')
add_arg('cutout_length',     int,   16,              "Cutout length.")
add_arg('drop_path_prob',    float, 0.3,             "Drop path probability.")
add_arg('save',              str,   'EXP',           "Experiment name.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('train_portion',     float, 0.5,             "Portion of training data.")
add_arg('arch_learning_rate',float, 3e-4,            "Learning rate for arch encoding.")
add_arg('arch_weight_decay', float, 1e-3,            "Weight decay for arch encoding.")
add_arg('image_shape',       str,   "3,32,32",       "input image size")
add_arg('with_mem_opt',      bool,  False,           "Whether to use memory optimization or not.")
#parser.add_argument('--unrolled', action='store_true', help='If set, use one-step unrolled validation loss.')
#yapf: enable

output_dir = './output/train_model/'
if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)

CIFAR10 = 50000

def main(args):
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    batch_size_per_device = args.batch_size // devices_num
    step_per_epoch = int(CIFAR10 * args.train_portion / args.batch_size)
    is_shuffle = True

    startup_prog = fluid.Program()
    data_prog = fluid.Program()
    test_prog = fluid.Program()

    image_shape = [int(m) for m in args.image_shape.split(",")]
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "begin construct graph")
    with fluid.unique_name.guard():
        with fluid.program_guard(data_prog, startup_prog):
            image_train = fluid.layers.data(name="image_train", shape=image_shape, dtype="float32")
            label_train = fluid.layers.data(name="label_train", shape=[1], dtype="int64")
            image_val = fluid.layers.data(name="image_val", shape=image_shape, dtype="float32")
            label_val = fluid.layers.data(name="label_val", shape=[1], dtype="int64")
            learning_rate = fluid.layers.cosine_decay(args.learning_rate, 8 * step_per_epoch,
                                    args.epochs)
            # Pytorch CosineAnnealingLR
            learning_rate = learning_rate / args.learning_rate * (args.learning_rate - args.learning_rate_min) + args.learning_rate_min

        unrolled_optim_prog, model_plus_prog, pos_grad_prog, model_minus_prog, neg_grad_prog, arch_optim_prog, fetch = architect.compute_unrolled_step(image_train, label_train, image_val,
                                    label_val, data_prog, startup_prog, learning_rate, args)

        train_prog = data_prog.clone()
        with fluid.program_guard(train_prog, startup_prog):
            logits, loss = model(image_train, label_train, args.init_channels,
                                             args.class_num, args.layers, name="model")
            top1 = fluid.layers.accuracy(input=logits, label=label_train, k=1)
            top5 = fluid.layers.accuracy(input=logits, label=label_train, k=5)
            test_prog = train_prog.clone(for_test=True)

            model_var = utility.get_parameters(train_prog.global_block().all_parameters(), 'model')[1]
            # update model_var with gradientclip
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0),
                param_list=[v.name for v in model_var])
            follower_opt = fluid.optimizer.MomentumOptimizer(learning_rate, args.momentum, regularization=fluid.regularizer.L2DecayRegularizer(args.weight_decay))
            follower_opt.minimize(loss, parameter_list=[v.name for v in model_var])

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "construct graph done")
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    train_reader, valid_reader = reader.train_search(batch_size=batch_size_per_device, train_portion=args.train_portion, is_shuffle=is_shuffle, args=args)
    train_batches = train_reader()
    valid_batches = valid_reader()

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4
    build_strategy = fluid.BuildStrategy()
    if args.with_mem_opt:
        learning_rate.persistable = True
        loss.persistable = True
        top1.persistable = True
        top5.persistable = True
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
    unrolled_optim_prog = fluid.CompiledProgram(unrolled_optim_prog).with_data_parallel(
                 loss_name=fetch[0].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    model_plus_prog = fluid.CompiledProgram(model_plus_prog).with_data_parallel(
                 loss_name=fetch[1].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    pos_grad_prog = fluid.CompiledProgram(pos_grad_prog).with_data_parallel(
                 loss_name=fetch[2].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    model_minus_prog = fluid.CompiledProgram(model_minus_prog).with_data_parallel(
                 loss_name=fetch[3].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    neg_grad_prog = fluid.CompiledProgram(neg_grad_prog).with_data_parallel(
                 loss_name=fetch[4].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    arch_optim_prog = fluid.CompiledProgram(arch_optim_prog).with_data_parallel(
                 loss_name=fetch[5].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
                 loss_name=loss.name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    print(time.asctime( time.localtime(time.time())), "compile graph done")

    def save_model(postfix, program):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    def genotype(epoch_id, train_batches):
        arch_names = utility.get_parameters(test_prog.global_block().all_parameters(), 'arch')[0]
        image_train, label_train = next(train_batches)
        feed = {"image_train": image_train, "label_train": label_train, "image_val": image_train, "label_val": label_train}
        arch_values = exe.run(test_prog, feed=feed, fetch_list=arch_names)
        # softmax
        arch_values = [np.exp(arch_v) / np.sum(np.exp(arch_v)) for arch_v in arch_values]
        alpha_normal = [i for i in zip(arch_names, arch_values) if 'weight1' in i[0]]
        alpha_reduce = [i for i in zip(arch_names, arch_values) if 'weight2' in i[0]]
        print('normal:')
        print(np.array([pair[1] for pair in sorted(alpha_normal, key=lambda i:int(i[0].split('_')[1]))]))
        print('reduce:')
        print(np.array([pair[1] for pair in sorted(alpha_reduce, key=lambda i:int(i[0].split('_')[1]))]))
        genotype = get_genotype(arch_names, arch_values)
        print("genotype={}".format(genotype))


    def valid(epoch_id, valid_batches, fetch_list):
        loss = utility.AvgrageMeter()
        top1 = utility.AvgrageMeter()
        top5 = utility.AvgrageMeter()
        for step_id in range(step_per_epoch):
            image_val, label_val = next(valid_batches)
            # use valid data to feed image_train and label_train
            feed = {"image_train": image_val, "label_train": label_val, "image_val": image_val, "label_val": label_val}
            loss_v, top1_v, top5_v = exe.run(test_prog, feed=feed, fetch_list=valid_fetch_list)
            loss.update(loss_v, args.batch_size)
            top1.update(top1_v, args.batch_size)
            top5.update(top5_v, args.batch_size)
            if step_id % args.report_freq == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                    "Valid Epoch {}, Step {}, loss {:.3f}, acc_1 {:.6f}, acc_5 {:.6f}"\
                    .format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[0]))
        return top1.avg[0]

    def train(epoch_id, train_batches, valid_batches, fetch_list):
        loss = utility.AvgrageMeter()
        top1 = utility.AvgrageMeter()
        top5 = utility.AvgrageMeter()
        for step_id in range(step_per_epoch):
            if args.profile:
                if epoch_id == 0 and step_id == 1:
                    profiler.start_profiler("All")
                elif epoch_id == 0 and step_id == 3:
                    profiler.stop_profiler("total", "/tmp/profile")
            image_train, label_train = next(train_batches)
            image_val, label_val = next(valid_batches)
            feed = {"image_train": image_train, "label_train": label_train, "image_val": image_val, "label_val": label_val}
            exe.run(unrolled_optim_prog, feed=feed)
            exe.run(model_plus_prog, feed=feed)
            exe.run(pos_grad_prog, feed=feed)
            exe.run(model_minus_prog, feed=feed)
            exe.run(neg_grad_prog, feed=feed)
            exe.run(arch_optim_prog, feed=feed)
            lr, loss_v, top1_v, top5_v = exe.run(train_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
            loss.update(loss_v, args.batch_size)
            top1.update(top1_v, args.batch_size)
            top5.update(top5_v, args.batch_size)
            if step_id % args.report_freq == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                    "Train Epoch {}, Step {}, Lr {:.8f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}"\
                    .format(epoch_id, step_id, lr[0], loss.avg[0], top1.avg[0], top5.avg[0]))
        return top1.avg[0]


    for epoch_id in range(args.epochs):
        # get genotype
        genotype(epoch_id, train_batches)
        train_fetch_list = [learning_rate, loss, top1, top5]
        train_top1 = train(epoch_id, train_batches, valid_batches, train_fetch_list)
        print("Epoch {}, train_acc {:.6f}".format(epoch_id, train_top1))
        valid_fetch_list = [loss, top1, top5]
        valid_top1 = valid(epoch_id, valid_batches, valid_fetch_list)
        print("Epoch {}, valid_acc {:.6f}".format(epoch_id, valid_top1))
        # (TODO)save model


if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)
