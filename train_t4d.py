# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
# coding :UTF-8
"""
Time:     2021/12/20 上午10:43
Author:   Jizhiwei
Version:  V 0.1
File:     train_t4d.py
Describe: Writen during my master's degree at ZJUT
"""
from __future__ import absolute_import, division, print_function

from trainer_t4d import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
