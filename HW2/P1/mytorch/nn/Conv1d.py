# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        b, c, n = A.shape
        Z = np.zeros((b, self.out_channels, n - self.kernel_size + 1)) # TODO

        for i in range(n - self.kernel_size + 1):
            # Z[:, :, i] = A[:, :, i:i + self.kernel_size] @ self.W + self.b 
            Z[:, :, i] = np.tensordot(A[:, :, i:i + self.kernel_size], self.W, axes=([1,2], [1,2])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        self.dLdW = np.zeros(self.W.shape)  # TODO
        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # TODO
        
        b, _, n = dLdZ.shape
        dLdA = np.zeros((b, self.in_channels, n + self.kernel_size - 1))  # TODO
        
        # perform operations in reverse order using np.tensordot
        for i in range(n):
            self.dLdW += np.tensordot(dLdZ[:, :, i], self.A[:, :, i:i + self.kernel_size], axes=([0], [0]))
            
            for j in range(self.kernel_size):
                # dLdA[:, :, i:i + self.kernel_size] += np.tensordot(dLdZ[:, :, i], self.W, axes=([1, 2], [0, 1]))
                dLdA[:, :, i + j] += np.tensordot(dLdZ[:, :, i], self.W[:, :, j], axes=([1], [0]))



        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant')# TODO

        # Call Conv1d_stride1
        conv_forward = self.conv1d_stride1.forward(A_padded)  # TODO

        # downsample
        Z = self.downsample1d.forward(conv_forward)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dldz_upsampled = self.downsample1d.backward(dLdZ)  # TODO

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv1d_stride1.backward(dldz_upsampled)  # TODO

        # Unpad the gradient
        # TODO
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded

        return dLdA