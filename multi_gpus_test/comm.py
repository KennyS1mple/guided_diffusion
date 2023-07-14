# CACIOUS CODING
# Data     : 7/6/23  4:21 PM
# File name: comm
# Desc     :

"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import logging
import numpy as np
import pickle
import torch
import torch.distributed as dist
import os

_LOCAL_PROCESS_GROUP = None


# 获得参与训练的进程总数，如N节点(每个2张GPU卡），则world_size = 2*N
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# 获取当前节点的编号，如参加训练的N张卡，主节点是0，其余的是1，2,......,N-1
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


# 获取当前节点下的单进程的gpu编号，这里获得的不是全局rank
def get_local_rank():
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0

    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


# 获取当前节点下的总进程数，即每台机器的进程个数
def get_local_size():
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def select_device(device=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available()

    cuda = not cpu and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')
