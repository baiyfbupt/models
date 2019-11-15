from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import sys
import logging
import paddle
import argparse
import functools
import paddle.fluid as fluid
sys.path.append("..")
import imagenet_reader as reader
import models
sys.path.append("../../")
from utility import add_arguments, print_arguments
from single_distiller import merge, l2_loss

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64*4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('total_images',     int,  1281167,              "Training image number.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('model',            str,  "MobileNet",          "Set the network to use.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('teacher_model',    str,  None,          "Set the teacher network to use.")
add_arg('teacher_pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('compress_config',  str,  None,                 "The config file for compression with yaml format.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def compress(args):
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    student_program = fluid.Program()
    s_startup = fluid.Program()

    with fluid.program_guard(student_program, s_startup):
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label],
            capacity=64,
            use_double_buffer=True,
            iterable=False)
        # model definition
        model = models.__dict__[args.model]()

        if args.model == 'ResNet34':
            model.prefix_name = 'res34'
            out = model.net(input=image, class_dim=args.class_dim)
        else:
            out = model.net(input=image, class_dim=args.class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    #print("="*50+"student_model_params"+"="*50)
    #for v in fluid.default_main_program().list_vars():
    #    print(v.name, v.shape)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    train_reader = paddle.batch(
        reader.train(), batch_size=args.batch_size, drop_last=True)
    train_feed_list = [('image', image.name), ('label', label.name)]
    train_fetch_list = [('loss', avg_cost.name)]

    places = fluid.cuda_places()
    data_loader.set_sample_list_generator(train_reader, places)

    teacher_model = models.__dict__[args.teacher_model]()
    # define teacher program
    teacher_program = fluid.Program()
    t_startup = fluid.Program()
    load_inference_model = True
    if not load_inference_model:
        teacher_scope = fluid.Scope()
        with fluid.scope_guard(teacher_scope):
            with fluid.program_guard(teacher_program, t_startup):
                image = fluid.layers.data(
                    name='xxx', shape=image_shape, dtype='float32')
                predict = teacher_model.net(image, class_dim=args.class_dim)
        #print("="*50+"teacher_model_params"+"="*50)
        #for v in teacher_program.list_vars():
        #    print(v.name, v.shape)
        #return

            exe.run(t_startup)
            assert args.teacher_pretrained_model and os.path.exists(
                args.teacher_pretrained_model
            ), "teacher_pretrained_model should be set when teacher_model is not None."

            def if_exist(var):
                return os.path.exists(
                    os.path.join(args.teacher_pretrained_model, var.name))

            fluid.io.load_vars(
                exe,
                args.teacher_pretrained_model,
                main_program=teacher_program,
                predicate=if_exist)
            #fluid.io.save_inference_model(dirname='./saved_for_inference', feeded_var_names=['xxx'], target_vars=[predict], executor=exe, main_program=teacher_program)
            #return

            teacher_program = teacher_program.clone(for_test=True)
    else:
        teacher_scope = fluid.global_scope()
        teacher_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(
            dirname='./saved_for_inference', executor=exe)

    data_name_map = {'xxx': 'image'}
    main = merge(
        teacher_program,
        student_program,
        data_name_map,
        place,
        teacher_scope=teacher_scope)
    #print("="*50+"teacher_vars"+"="*50)
    #for v in teacher_program.list_vars():
    #    if '_generated_var' not in v.name and 'fetch' not in v.name and 'feed' not in v.name:
    #        print(v.name, v.shape)
    #return
    with fluid.program_guard(student_program, s_startup):
        dist_loss = l2_loss("teacher_fc_1.tmp_0", "fc_0.tmp_0", main)
        loss = avg_cost + dist_loss
        opt = fluid.optimizer.Adam(
            regularization=fluid.regularizer.L2Decay(4e-5))
        opt.minimize(loss)
    exe.run(s_startup)

    #feeder = fluid.DataFeeder(
    #    feed_list=['image', 'label'], place=place, program=main)
    iterable_reader = False
    if iterable_reader:
        for step_id, data in enumerate(data_loader):
            #data = feeder.feed(data)

            loss_1, loss_2 = exe.run(
                main, feed=data, fetch_list=[avg_cost.name, dist_loss.name])
            _logger.info("step {} class loss {:.6f} dist loss {:.6f}".format(
                step_id, loss_1[0], loss_2[0]))
    else:
        step_id = 0
        data_loader.start()
        while True:
            loss_1, loss_2 = exe.run(
                main, fetch_list=[avg_cost.name, dist_loss.name])
            _logger.info("step {} class loss {:.6f} dist loss {:.6f}".format(
                step_id, loss_1[0], loss_2[0]))
            step_id += 1


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
