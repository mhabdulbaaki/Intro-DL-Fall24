import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        self.A = A
        batch_size, in_channels, in_height, in_width = A.shape
        out_height = in_height - self.kernel + 1
        out_width = in_width - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                Z[:, :, i, j] = np.max(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        batch_size, in_channels, out_height, out_width = dLdZ.shape
        _, _, in_height, in_width = self.A.shape

        dLdA = np.zeros_like(self.A)
        
        for i in range(out_height):
            for j in range(out_width):
                window = self.A[:, :, i:i+self.kernel, j:j+self.kernel]
                mask = (window == np.max(window, axis=(2, 3))[:, :, np.newaxis, np.newaxis])
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += mask * dLdZ[:, :, i, j][:, :, np.newaxis, np.newaxis]
        
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, in_height, in_width = A.shape
        out_height = in_height - self.kernel + 1
        out_width = in_width - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                Z[:, :, i, j] = np.mean(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, in_channels, out_height, out_width = dLdZ.shape
        _, _, in_height, in_width = self.A.shape

        dLdA = np.zeros_like(self.A)
        
        for i in range(out_height):
            for j in range(out_width):
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += dLdZ[:, :, i, j][:, :, np.newaxis, np.newaxis] / (self.kernel * self.kernel)
        
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ_upsampled)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z = self.meanpool2d_stride1.forward(A)
        
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ_upsampled)
        
        return dLdA
