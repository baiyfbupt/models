"""LightNASNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = ['LightNASNet']

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class LightNASNet(object):
    """LightNASNet."""

    def __init__(self):
        self.params = train_parameters

    def net(self, input, bottleneck_params_list=None, class_dim=1000,
            scale=1.0):
        """Build network.

        Args:
            input: Variable, input.
            class_dim: int, class dim.
            scale: float, scale.

        Returns:
            Variable, network output.
        """
        if bottleneck_params_list is None:
            # MobileNetV2
            # bottleneck_params_list = [
            #     (1, 16, 1, 1, 3, 1, 0),
            #     (6, 24, 2, 2, 3, 1, 0),
            #     (6, 32, 3, 2, 3, 1, 0),
            #     (6, 64, 4, 2, 3, 1, 0),
            #     (6, 96, 3, 1, 3, 1, 0),
            #     (6, 160, 3, 2, 3, 1, 0),
            #     (6, 320, 1, 1, 3, 1, 0),
            # ]
            bottleneck_params_list = [
                (1, 16, 1, 1, 3, 1, 0),
                (3, 24, 3, 2, 3, 1, 0),
                (3, 40, 3, 2, 5, 1, 0),
                (6, 80, 3, 2, 5, 1, 0),
                (6, 96, 2, 1, 3, 1, 0),
                (6, 192, 4, 2, 5, 1, 0),
                (6, 320, 1, 1, 3, 1, 0),
            ]

        #conv1
        input = self.conv_bn_layer(
            input,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            name='conv1_1')

        # bottleneck sequences
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s, k, ifshortcut, ifse = layer_setting
            i += 1
            input = self.invresi_blocks(
                input=input,
                in_c=in_c,
                t=t,
                c=int(c * scale),
                n=n,
                s=s,
                k=k,
                ifshortcut=ifshortcut,
                ifse=ifse,
                name='conv' + str(i))
            in_c = int(c * scale)

        #last_conv
        input = self.conv_bn_layer(
            input=input,
            num_filters=int(1280 * scale) if scale > 1.0 else 1280,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            name='conv9')

        input = fluid.layers.pool2d(
            input=input,
            pool_size=7,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)

        input = fluid.layers.dropout(input, 0.2)

        init_range = 1.0 / np.sqrt(class_dim)
        output = fluid.layers.fc(
            input=input,
            size=class_dim,
            param_attr=ParamAttr(
                name='fc10_weights',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_range, high=init_range)),
            #param_attr=fluid.initializer.UniformInitializer(low=-init_range, high=init_range),
            bias_attr=False)
        return output

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      name=None,
                      use_cudnn=True):
        """Build convolution and batch normalization layers.

        Args:
            input: Variable, input.
            filter_size: int, filter size.
            num_filters: int, number of filters.
            stride: int, stride.
            padding: int, padding.
            num_groups: int, number of groups.
            if_act: bool, whether using activation.
            name: str, name.
            use_cudnn: bool, whether use cudnn.

        Returns:
            Variable, layers output.
        """
        input_size = input.shape[2]
        out_size = math.ceil(input_size / stride)
        padding_num = (out_size - 1) * stride + filter_size - input_size
        pad_left = int(math.floor(padding_num / 2))
        pad_right = int(padding_num - pad_left)
        input = fluid.layers.pad2d(
            input=input, paddings=[pad_left, pad_right, pad_left, pad_right])

        fan_out = int(filter_size * filter_size * num_filters)
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=0,
            groups=num_groups,
            use_cudnn=True,
            #param_attr=ParamAttr(name=name + '_weights'),
            #param_attr=fluid.initializer.Constant(value=0.1),
            param_attr=ParamAttr(
                name=name + '_weights',
                initializer=fluid.initializer.Normal(
                    loc=0.0, scale=np.sqrt(2.0 / fan_out))),
            #param_attr=fluid.initializer.Normal(loc=0.0, scale=np.sqrt(2.0 / fan_out)),
            bias_attr=False)

        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            momentum=0.99,
            epsilon=1e-3,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

        if if_act:
            return fluid.layers.relu(bn)
        else:
            return bn

    def shortcut(self, input, data_residual):
        """Build shortcut layer.

        Args:
            input: Variable, input.
            data_residual: Variable, residual layer.

        Returns:
            Variable, layer output.
        """
        return fluid.layers.elementwise_add(input, data_residual)

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        """Build squeeze excitation layers.

        Args:
            input: Variable, input.
            num_channels: int, number of channels.
            reduction_ratio: float, reduction ratio.
            name: str, name.

        Returns:
            Variable, layers output.
        """
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale

    def inverted_residual_unit(self,
                               input,
                               num_in_filter,
                               num_filters,
                               ifshortcut,
                               ifse,
                               stride,
                               filter_size,
                               expansion_factor,
                               reduction_ratio=4,
                               name=None):
        """Build inverted residual unit.

        Args:
            input: Variable, input.
            num_in_filter: int, number of in filters.
            num_filters: int, number of filters.
            ifshortcut: bool, whether using shortcut.
            stride: int, stride.
            filter_size: int, filter size.
            padding: int, padding.
            expansion_factor: float, expansion factor.
            name: str, name.

        Returns:
            Variable, layers output.
        """
        num_expfilter = int(round(num_in_filter * expansion_factor))

        x = input
        if int(expansion_factor) != 1:
            x = self.conv_bn_layer(
                input=x,
                num_filters=num_expfilter,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name=name + '_expand')

        bottleneck_conv = self.conv_bn_layer(
            input=x,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) / 2),
            num_groups=num_expfilter,
            if_act=True,
            name=name + '_dwise',
            use_cudnn=False)

        linear_out = self.conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=False,
            name=name + '_linear')
        out = linear_out

        if ifshortcut:
            out = self.shortcut(input=input, data_residual=out)

        if ifse:
            scale = self.squeeze_excitation(
                input=linear_out,
                num_channels=num_filters,
                reduction_ratio=reduction_ratio,
                name=name + '_fc')
            out = fluid.layers.elementwise_add(x=out, y=scale, act='relu')
        return out

    def invresi_blocks(self,
                       input,
                       in_c,
                       t,
                       c,
                       n,
                       s,
                       k,
                       ifshortcut,
                       ifse,
                       name=None):
        """Build inverted residual blocks.

        Args:
            input: Variable, input.
            in_c: int, number of in filters.
            t: float, expansion factor.
            c: int, number of filters.
            n: int, number of layers.
            s: int, stride.
            k: int, filter size.
            ifshortcut: bool, if adding shortcut layers or not.
            ifse: bool, if adding squeeze excitation layers or not.
            name: str, name.

        Returns:
            Variable, layers output.
        """
        first_block = self.inverted_residual_unit(
            input=input,
            num_in_filter=in_c,
            num_filters=c,
            ifshortcut=False,
            ifse=ifse,
            stride=s,
            filter_size=k,
            expansion_factor=t,
            name=name + '_1')

        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block = self.inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=ifshortcut,
                ifse=ifse,
                stride=1,
                filter_size=k,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block
