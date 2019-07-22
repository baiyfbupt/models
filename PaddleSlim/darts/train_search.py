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
import math
import time
import paddle.fluid as fluid
import reader

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel',          bool,  True,            "Whether use multi-GPU/threads or not.")
add_arg('data',              str,   '../data',       "The dir of dataset.")
add_arg('batch_size',        int,   16,              "Minibatch size.")
add_arg('learning_rate',     float, 0.025,           "The start learning rate.")
add_arg('learning_rate_min', float, 0.001,           "The min learning rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('weight_decay',      float, 3e-4,            "Weight_decay.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   50,              "Epoch number.")
add_arg('init_channels',     int,   16,              "Init channel number.")
add_arg('layers',            int,   8,               "Total number of layers.")
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

output_dir='/outputs/train_model/'
if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)

CLASS_NUM = 10

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
            image_train, label_train, image_val, label_val = fluid.layers.read_file(py_reader)
            fetches = []

def main(args):
    batch_size = args.batch_size
    epochs = args.epochs
    data_dir = args.data
    model_save_dir = args.model_save_dir
    with_memory_optimization = args.with_mem_opt

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    batch_size_per_device = batch_size // devices_num
    num_workers = 8
    is_shuffle = True

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    train_py_reader, val_py_reader, fetches, loss = build_program(
        main_prog = train_prog,
        startup_prog = startup_prog,
        args = args
    )


    if with_memory_optimization:
        fluid.memory_optimize(train_prog)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)










if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    check_cuda(args.use_gpu)

    main(args)
