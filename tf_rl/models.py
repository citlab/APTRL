"""
based on https://github.com/nivwusquorum/tensorflow-deepq under
the following license:

The MIT License (MIT)

Copyright (c) 2015 Szymon Sidor

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.
"""

from ascar_logging import logger
import math
import tensorflow.compat.v1 as tf

from .utils import base_name


class Layer(object):
    def __init__(self, input_sizes, output_size, scope):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        # Specify input and output dimension of layer
        self.input_sizes = input_sizes
        self.output_size = output_size
        self.scope       = scope or "Layer"

        with tf.variable_scope(self.scope):
            # Initialize weight of each input
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                W_name = "W_%d" % (input_idx,)
                # random initial weight
                W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
                W_var = tf.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
                self.Ws.append(W_var)
            self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
        with tf.variable_scope(self.scope):
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

    def variables(self):
        return [self.b] + self.Ws

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                                initializer=lambda x, dtype=tf.float32, partition_info=None: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)


class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None):
        # input and hidden layer dimension
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        # activation function
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        # Scope for using with tf variable
        self.scope = scope or "MLP"

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.variable_scope(self.scope):
            if given_layers is not None:
                self.input_layer = given_layers[0]
                self.layers      = given_layers[1:]
            else:
                self.input_layer = Layer(input_sizes, hiddens[0], scope="input_layer")
                self.layers = []

                for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                        self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d" % (l_idx,)))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.variable_scope(self.scope):
            # Run tanh activation function of first layer
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                # In last layer run identity function
                hidden = nonlinearity(layer(hidden))
            return hidden

    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
        return MLP(self.input_sizes, self.hiddens, nonlinearities, scope=scope,
                given_layers=given_layers)


class ConvLayer(object):
    def __init__(self, filter_H, filter_W,
                    in_C, out_C,
                    stride=(1,1),
                    scope="Convolution"):
        self.filter_H, self.filter_W = filter_H, filter_W
        self.in_C,     self.out_C    = in_C,     out_C
        self.stride                  = stride
        self.scope                   = scope

        with tf.variable_scope(self.scope):
            input_size = filter_H * filter_W * in_C
            W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(input_size),
                        1.0 / math.sqrt(input_size))
            self.W = tf.get_variable('W',
                    (filter_H, filter_W, in_C, out_C),
                    initializer=W_initializer)
            self.b = tf.get_variable('b',
                    (out_C),
                    initializer=tf.constant_initializer(0))

    def __call__(self, X):
        with tf.variable_scope(self.scope):
            return tf.nn.conv2d(X, self.W,
                                strides=[1] + list(self.stride) + [1],
                                padding='SAME') + self.b

    def variables(self):
        return [self.W, self.b]

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return ConvLayer(self.filter_H, self.filter_W, self.in_C, self.out_C, self.stride, scope=sc)

class SeqLayer(object):
    def __init__(self, layers, scope='seq_layer'):
        self.scope = scope
        self.layers = layers

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def variables(self):
        return sum([l.variables() for l in self.layers], [])

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(self.scope):
            copied_layers = [layer.copy() for layer in self.layers]
            return SeqLayer(copied_layers, scope=scope)


class LambdaLayer(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def variables(self):
        return []

    def copy(self):
        return LambdaLayer(self.f)
