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
import reader
import utility
import architect
from model_search import model

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel',          bool,  True,            "Whether use multi-GPU/threads or not.")
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
#parser.add_argument('--unrolled', action='store_true', help='If set, use one-step unrolled validation loss.')
#yapf: enable

output_dir = './outputs/train_model/'
if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)

CIFAR10 = 50000

def main(args):
    model_save_dir = args.model_save_dir

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    batch_size_per_device = args.batch_size // devices_num
    step_per_epoch = int(CIFAR10 * args.train_portion / args.batch_size)
    is_shuffle = True

    startup_prog = fluid.Program()
    data_prog = fluid.Program()
    test_prog = fluid.Program()

    image_shape = [int(m) for m in args.image_shape.split(",")]
    print(time.asctime( time.localtime(time.time())), "begin construct graph")
    with fluid.unique_name.guard():
        with fluid.program_guard(data_prog, startup_prog):
            image_train = fluid.layers.data(name="image_train", shape=image_shape, dtype="float32")
            label_train = fluid.layers.data(name="label_train", shape=[1], dtype="int64")
            image_val = fluid.layers.data(name="image_val", shape=image_shape, dtype="float32")
            label_val = fluid.layers.data(name="label_val", shape=[1], dtype="int64")
            learning_rate = fluid.layers.cosine_decay(args.learning_rate, step_per_epoch,
                                    args.epochs)
            # Pytorch CosineAnnealingLR
            learning_rate = learning_rate / args.learning_rate * (args.learning_rate - args.learning_rate_min) + args.learning_rate_min

        unrolled_optim_prog, model_plus_prog, pos_grad_prog, model_minus_prog, neg_grad_prog, arch_optim_prog, fetch = architect.compute_unrolled_step(image_train, label_train, image_val,
                                    label_val, data_prog, startup_prog, learning_rate, args)

        train_prog = data_prog.clone()
        with fluid.program_guard(train_prog, startup_prog):
            train_logits, train_loss = model(image_train, label_train, args.init_channels,
                                             args.class_num, args.layers, name="model")
            train_top1 = fluid.layers.accuracy(input=train_logits, label=label_train, k=1)
            train_top5 = fluid.layers.accuracy(input=train_logits, label=label_train, k=5)

            model_var = utility.get_parameters(train_prog.global_block().all_parameters(), 'model')[1]
            # update model_var with gradientclip
            model_grads = fluid.gradients(train_loss, model_var)
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0),
                param_list=model_var)
            model_params_grads = list(zip(model_var, model_grads))
            follower_opt = fluid.optimizer.MomentumOptimizer(learning_rate, args.momentum, regularization=fluid.regularizer.L2DecayRegularizer(args.weight_decay))
            follower_opt.apply_gradients(model_params_grads)

    print(time.asctime( time.localtime(time.time())), "construct graph done")
    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    print(time.asctime( time.localtime(time.time())), "init done")

    batches = reader.train_val(args, batch_size_per_device, args.train_portion, is_shuffle)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4
    build_strategy = fluid.BuildStrategy()
    if args.with_mem_opt:
        train_loss.persistable = True
        train_top1.persistable = True
        train_top5.persistable = True
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
    '''
    train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
                 loss_name=train_loss.name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    pos_grad_prog = fluid.CompiledProgram(pos_grad_prog).with_data_parallel(
                 loss_name=fetch[0].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    neg_grad_prog = fluid.CompiledProgram(neg_grad_prog).with_data_parallel(
                 loss_name=fetch[1].name,
                 build_strategy=build_strategy,
                 exec_strategy=exec_strategy)
    print(time.asctime( time.localtime(time.time())), "compile graph done")
    '''
    def save_model(postfix, program):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    def valid(epoch_id, batches, test_prog):
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                image = fluid.layers.data(name="image", shape=image_shape, dtype="float32")
                label = fluid.layers.data(name="label", shape=[1], dtype="int64")
                valid_logits, valid_loss = model(image, label, args.init_channels, args.class_num, args.layers, name='model')
                valid_top1 = fluid.layers.accuracy(input=valid_logits, label=label, k=1)
                valid_top5 = fluid.layers.accuracy(input=valid_logits, label=label, k=5)
        valid_fetch_list = [valid_loss, valid_top1, valid_top5]
        loss = utility.AvgrageMeter()
        top1 = utility.AvgrageMeter()
        top5 = utility.AvgrageMeter()
        for step_id in range(step_per_epoch):
            _, _, image_val, label_val = next(batches())
            feed = {"image": image_val, "label": label_val}
            loss_v, top1_v, top5_v = exe.run(test_prog, feed=feed, fetch_list=valid_fetch_list)
            loss.update(np.array(loss_v), args.batch_size)
            top1.update(np.array(top1_v), args.batch_size)
            top5.update(np.array(top5_v), args.batch_size)
            print(time.asctime(time.localtime(time.time())), "Valid Epoch {}, Step {}, loss {:.3f}, acc_1 {:.6f}, acc_5 {:.6f}".format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[0]))
        print(time.asctime(time.localtime(time.time())), "Epoch {}, top1 {:.6f}, top5 {:.6f}".format(epoch_id, top1.avg[0], top5.avg[0]))


    fetch_list = [learning_rate, train_loss, train_top1, train_top5]
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for epoch_id in range(args.epochs):
        for step_id in range(step_per_epoch):
            image_train, label_train, image_val, label_val = next(batches())
            feed = {"image_train": image_train, "label_train": label_train, "image_val": image_val, "label_val": label_val}
            _ = exe.run(unrolled_optim_prog, feed=feed, fetch_list=[fetch[0].name])
            _ = exe.run(model_plus_prog, feed=feed, fetch_list=[fetch[1].name])
            _ = exe.run(pos_grad_prog, feed=feed, fetch_list=[fetch[2].name])
            _ = exe.run(model_minus_prog, feed=feed, fetch_list=[fetch[3].name])
            _ = exe.run(neg_grad_prog, feed=feed, fetch_list=[fetch[4].name])
            _ = exe.run(arch_optim_prog, feed=feed, fetch_list=[fetch[5].name])
            lr_v, loss_v, top1_v, top5_v = exe.run(train_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
            loss.update(np.array(loss_v), args.batch_size)
            top1.update(np.array(top1_v), args.batch_size)
            top5.update(np.array(top5_v), args.batch_size)
            lr = np.array(lr_v)
            print(time.asctime(time.localtime(time.time())), "Train Epoch {}, Step {}, Lr {:.3f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".format(epoch_id, step_id, lr[0], loss.avg[0], top1.avg[0], top5.avg[0]))
        valid(epoch_id, batches, test_prog)



if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)
