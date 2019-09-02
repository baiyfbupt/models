# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6], )

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6])

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('skip_connect', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

pass0 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2),
            ('dil_conv_5x5', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2),
            ('skip_connect', 1), ('skip_connect', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
            ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3),
            ('skip_connect', 4), ('skip_connect', 3)],
    reduce_concat=[2, 3, 4, 5])
pass1 = Genotype(
    normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])
pass2 = Genotype(
    normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 0), ('sep_conv_3x3', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])
pass3 = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 0), ('sep_conv_3x3', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])
pass4 = Genotype(
    normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass5 = Genotype(
    normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass6 = Genotype(
    normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass7 = Genotype(
    normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass8 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass9 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass10 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass11 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass12 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass13 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass14 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 4), ('max_pool_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass15 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass16 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass17 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass18 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass19 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass20 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 2)],
    reduce_concat=[2, 3, 4, 5])
pass21 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass22 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass23 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass24 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass25 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass26 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass27 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass28 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass29 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass30 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 3)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass31 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass32 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass33 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass34 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass35 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass36 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass37 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass38 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass39 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass40 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass41 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])
pass42 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass43 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass44 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass45 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass46 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass47 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass48 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
pass49 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5])
