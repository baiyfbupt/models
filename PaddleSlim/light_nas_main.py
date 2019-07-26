"""Light Neural Architecture Search main process to get reward."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import json
import math
import time
import argparse
import subprocess
import functools
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
from paddle.fluid.contrib.slim import Context
from paddle.fluid.contrib.slim.graph import GraphWrapper

import reader
import models
from utils.learning_rate import cosine_decay
from utils.learning_rate import cosine_decay_with_warmup
from utils.fp16_utils import create_master_params_grads, master_param_to_train_param
from utility import add_arguments

IMAGENET1000 = 1281167
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('bottleneck_params_list', str, '[1,16,1,1,3,1,0,6,24,2,2,3,1,0,'
        '6,32,3,2,3,1,0,6,64,4,2,3,1,0,6,96,3,1,3,1,0,6,160,3,2,3,1,0,'
        '6,320,1,1,3,1,0]', "Network architecture.")
add_arg('target_latency', float, 629145600, "Target latency.")
add_arg('metric', str, 'flops', "Metric for latency: flops/time.")
add_arg('batch_size', int, 4, "Minibatch size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('total_images', int, 1281167, "Training image number.")
add_arg('num_epochs', int, 1, "number of epochs.")
add_arg('class_dim', int, 1000, "Class number.")
add_arg('image_shape', str, "3,224,224", "input image size")
add_arg('model_save_dir', str, "output", "model save directory")
add_arg('with_mem_opt', bool, True,
        "Whether to use memory optimization or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('checkpoint', str, None, "Whether to resume checkpoint.")
add_arg('lr', float, 0.1, "set learning rate.")
add_arg('lr_strategy', str, "cosine_decay",
        "Set the learning rate decay strategy.")
add_arg('model', str, "LightNASNet", "Set the network to use.")
add_arg('enable_ce', bool, False,
        "If set True, enable continuous evaluation job.")
add_arg('data_dir', str, "./data/ILSVRC2012", "The ImageNet dataset root dir.")
add_arg('fp16', bool, False, "Enable half precision training with fp16.")
add_arg('scale_loss', float, 1.0, "Scale loss for fp16.")
add_arg('l2_decay', float, 4e-5, "L2_decay parameter.")
add_arg('momentum_rate', float, 0.9, "momentum_rate.")
add_arg('use_ngraph', bool, False, "Whether to use NGraph engine or not.")


def optimizer_setting(params):
    """optimizer setting.

    Args:
        params: dict, params.
    """
    ls = params["learning_strategy"]
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)
        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "cosine_warmup_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        step = int(math.ceil(float(total_images) / batch_size))
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay_with_warmup(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "linear_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        num_epochs = params["num_epochs"]
        start_lr = params["lr"]
        end_lr = 0
        total_step = int((total_images / batch_size) * num_epochs)
        lr = fluid.layers.polynomial_decay(
            start_lr, total_step, end_lr, power=1)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "adam":
        lr = params["lr"]
        optimizer = fluid.optimizer.Adam(learning_rate=lr)
    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    return optimizer


def net_config(image, label, model, args):
    """net config.

    Args:
        image: Variable, image.
        label: Variable, label.
        model: str, model name.
        args: args.

    Returns:
        tuple, (avg_cost, acc_top1, acc_top5)
    """
    class_dim = args.class_dim
    bottleneck_params_list = json.loads(args.bottleneck_params_list)
    bottleneck_params_list = [
        bottleneck_params_list[i:i + 7]
        for i in range(0, len(bottleneck_params_list), 7)
    ]
    out = model.net(input=image,
                    bottleneck_params_list=bottleneck_params_list,
                    class_dim=class_dim)
    cost, pred = fluid.layers.softmax_with_cross_entropy(
        out, label, return_softmax=True)
    fluid.layers.Print(input=pred, message="\npred:", summarize=200)
    fluid.layers.Print(input=cost, message="\ncost:", summarize=200)
    if args.scale_loss > 1:
        avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
    else:
        avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)
    return avg_cost, acc_top1, acc_top5


def build_program(is_train, main_prog, startup_prog, args):
    """build program.

    Args:
        is_train: bool, whether is training.
        main_prog: main program.
        startup_prog: startup program.
        args: args.

    Returns:
        tuple,
            (py_reader, avg_cost, acc_top1, acc_top5, global_lr) if is_train
            (py_reader, avg_cost, acc_top1, acc_top5) else
    """
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    model = models.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            if args.fp16:
                image = fluid.layers.cast(image, "float16")
            avg_cost, acc_top1, acc_top5 = net_config(image, label, model, args)
            avg_cost.persistable = True
            acc_top1.persistable = True
            acc_top5.persistable = True
            if is_train:
                params = model.params
                params["total_images"] = args.total_images
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["learning_strategy"]["batch_size"] = args.batch_size
                params["learning_strategy"]["name"] = args.lr_strategy
                params["l2_decay"] = args.l2_decay
                params["momentum_rate"] = args.momentum_rate
                optimizer = optimizer_setting(params)
                if args.fp16:
                    params_grads = optimizer.backward(avg_cost)
                    master_params_grads = create_master_params_grads(
                        params_grads, main_prog, startup_prog, args.scale_loss)
                    optimizer.apply_gradients(master_params_grads)
                    master_param_to_train_param(master_params_grads,
                                                params_grads, main_prog)
                else:
                    optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()
    if is_train:
        return py_reader, avg_cost, acc_top1, acc_top5, global_lr
    else:
        return py_reader, avg_cost, acc_top1, acc_top5


def get_device_num(use_gpu):
    """Get device number.

    Args:
        use_gpu: bool, whether use gpu.

    Returns:
        int, device number.
    """
    if use_gpu:
        visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
        if visible_device:
            device_num = len(visible_device.split(','))
        else:
            device_num = subprocess.check_output(['nvidia-smi',
                                                  '-L']).decode().count('\n')
    else:
        device_num = 1
    return device_num


def get_flops(test_prog, place):
    """Get flops.

    Args:
        test_prog: test program.
        place: place.

    Returns:
        flops: long, flops.
        numel_params: numpy.int64, number of parameters.
    """
    eval_graph = GraphWrapper(test_prog)
    context = Context(
        place=place, scope=fluid.global_scope(), eval_graph=eval_graph)
    flops = context.eval_graph.flops()
    numel_params = context.eval_graph.numel_params()
    return flops, numel_params


def train(args):
    """train.

    Args:
        args: args.
    """
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    model_save_dir = args.model_save_dir
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    if args.enable_ce:
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000
    train_py_reader, train_cost, train_acc1, train_acc5, global_lr = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    test_py_reader, test_cost, test_acc1, test_acc5 = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = test_prog.clone(for_test=True)
    if with_memory_optimization:
        fluid.memory_optimize(train_prog)
        fluid.memory_optimize(test_prog)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    if checkpoint:
        fluid.io.load_persistables(exe, checkpoint, main_program=train_prog)
    elif pretrained_model:

        def if_exist(var):
            """Check whether var exists.

            Args:
                var: Variable.

            Returns:
                bool, whether exist.
            """
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)
    device_num = get_device_num(args.use_gpu)
    train_batch_size = args.batch_size / device_num
    test_batch_size = 16
    if not args.enable_ce:
        train_reader = paddle.batch(
            reader.train(), batch_size=train_batch_size, drop_last=True)
        test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)
    else:
        # use flowers dataset for CE and set use_xmap False to avoid disorder data
        # but it is time consuming. For faster speed, need another dataset.
        import random
        random.seed(0)
        np.random.seed(0)
        train_reader = paddle.batch(
            flowers.train(use_xmap=False),
            batch_size=train_batch_size,
            drop_last=True)
        test_reader = paddle.batch(
            flowers.test(use_xmap=False), batch_size=test_batch_size)
    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)
    # use_ngraph = os.getenv('FLAGS_use_ngraph')
    use_ngraph = args.use_ngraph
    if not use_ngraph:
        train_exe = fluid.ParallelExecutor(
            main_program=train_prog,
            use_cuda=bool(args.use_gpu),
            loss_name=train_cost.name)
    else:
        train_exe = exe
    train_fetch_list = [
        train_cost.name, train_acc1.name, train_acc5.name, global_lr.name
    ]
    test_fetch_list = [test_cost.name, test_acc1.name, test_acc5.name]
    params = models.__dict__[args.model]().params
    for pass_id in range(params["num_epochs"]):
        train_py_reader.start()
        train_info = [[], [], []]
        test_info = [[], [], []]
        train_time = []
        batch_id = 0
        try:
            while True:
                t1 = time.time()
                if use_ngraph:
                    loss, acc1, acc5, lr = train_exe.run(
                        train_prog, fetch_list=train_fetch_list)
                else:
                    loss, acc1, acc5, lr = train_exe.run(
                        fetch_list=train_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(np.array(loss))
                acc1 = np.mean(np.array(acc1))
                acc5 = np.mean(np.array(acc5))
                train_info[0].append(loss)
                train_info[1].append(acc1)
                train_info[2].append(acc5)
                lr = np.mean(np.array(lr))
                train_time.append(period)
                batch_id += 1
        except fluid.core.EOFException:
            train_py_reader.reset()
        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        train_speed = np.array(train_time).mean() / (train_batch_size *
                                                     device_num)
        test_py_reader.start()
        test_batch_id = 0
        test_time = []
        try:
            while True:
                t1 = time.time()
                loss, acc1, acc5 = exe.run(program=test_prog,
                                           fetch_list=test_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(loss)
                acc1 = np.mean(acc1)
                acc5 = np.mean(acc5)
                test_info[0].append(loss)
                test_info[1].append(acc1)
                test_info[2].append(acc5)
                test_time.append(period)
                test_batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()
        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()
        test_acc1 *= 100
        period = np.array(test_time[1:]).mean() * 1000
        flops, numel_params = get_flops(test_prog, place)
        if args.metric == 'flops':
            reward = test_acc1 if flops <= args.target_latency else 0
        else:
            reward = test_acc1 if period <= args.target_latency else 0
        print('{} {}% {}ms flops: {} #params: {}'.format(
            reward, test_acc1, period, flops, numel_params))
        sys.stdout.flush()
        model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path, main_program=train_prog)
        # This is for continuous evaluation only
        if args.enable_ce and pass_id == args.num_epochs - 1:
            if device_num == 1:
                # Use the mean cost/acc for training
                print("kpis train_cost %s" % train_loss)
                print("kpis train_acc_top1 %s" % train_acc1)
                print("kpis train_acc_top5 %s" % train_acc5)
                # Use the mean cost/acc for testing
                print("kpis test_cost %s" % test_loss)
                print("kpis test_acc_top1 %s" % test_acc1)
                print("kpis test_acc_top5 %s" % test_acc5)
                print("kpis train_speed %s" % train_speed)
            else:
                # Use the mean cost/acc for training
                print("kpis train_cost_card%s %s" % (device_num, train_loss))
                print("kpis train_acc_top1_card%s %s" %
                      (device_num, train_acc1))
                print("kpis train_acc_top5_card%s %s" %
                      (device_num, train_acc5))
                # Use the mean cost/acc for testing
                print("kpis test_cost_card%s %s" % (device_num, test_loss))
                print("kpis test_acc_top1_card%s %s" % (device_num, test_acc1))
                print("kpis test_acc_top5_card%s %s" % (device_num, test_acc5))
                print("kpis train_speed_card%s %s" % (device_num, train_speed))


def main():
    """main.
    """
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
