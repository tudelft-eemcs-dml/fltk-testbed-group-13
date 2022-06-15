import collections
import logging
import math
import sys
import copy

import torch
import torch.distributed as dist
import functools
import numpy as np
from scipy import sparse
import mxnet as mx

def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def communicate(tensors, communication_op):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        t.set_(f)


def threshold_to_value(array, lower_bound, upper_bound, value=0):
    """
        changing values within a range to a single value
        :param array: numpy array
        :param lower_bound: lower bound of the threshold
        :param upper_bound: upper bound of the threshold
        :param value: value to change the range to; default = 0
        :return: numpy array where all values within the threshold are change to the given value
    """
    array = np.array(array)
    array[lower_bound < array < upper_bound] = value
    return array


def compress_array_to_buffer(array):
    """
        converting numpy array to bytes
        :param array: numpy array
        :return: buffer
    """
    return np.array(array).tobytes()


def decompress_array_from_buffer(received_buffer):
    """
       converting buffer to numpy array
       :param received_buffer: buffer of bytes
       :return: numpy array with dtype='float32'
    """
    return np.frombuffer(received_buffer, dtype='float32', count=-1)


def convert_array_to_sparse(array):
    """
        converting numpy array to sparse matrix
        :param array: numpy array
        :return: scipy sparse matrix
    """
    return sparse.csr_matrix(np.array(array))


def convert_sparse_to_array(matrix):
    """
       converting sparse matrix to numpy array
       :param matrix: scpy sparse matrix
       :return: numpy array
     """
    return np.array(matrix.toarray())


def quantize_float32_to_int8(data):
    """
       quantize float32 array to int8
       :param data: mx.ndarray of type float32
       :return: mx.ndarray of type int8
    """

    min_range = mx.nd.min(data)
    max_range = mx.nd.max(data)

    qdata, min_val, max_val = mx.nd.contrib.quantize(data, min_range, max_range, out_type='int8')

    return qdata


def dequantize_int8_to_float32(qdata, min_value, max_value):
    """
       dequantize int8 array to float32
       :param data: mx.ndarray of type int8
       :param min_value: min value of the original array
       :param max_value: max value of the original array
       :return: mx.ndarray of type float32
    """

    qdata = mx.nd.array(qdata, dtype=np.int8)
    min_range = mx.nd.array([min_value], dtype=np.float32)
    max_range = mx.nd.array([max_value], dtype=np.float32)

    return mx.nd.contrib.dequantize(qdata, min_range, max_range, out_type='float32')