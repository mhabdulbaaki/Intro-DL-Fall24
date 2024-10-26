import numpy as np
from resampling import *


# class Conv2d_stride1():
#     def __init__(self, in_channels, out_channels,
#                  kernel_size, weight_init_fn=None, bias_init_fn=None):

#         # Do not modify this method

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size

#         if weight_init_fn is None:
#             self.W = np.random.normal(
#                 0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
#         else:
#             self.W = weight_init_fn(
#                 out_channels,
#                 in_channels,
#                 kernel_size,
#                 kernel_size)

#         if bias_init_fn is None:
#             self.b = np.zeros(out_channels)
#         else:
#             self.b = bias_init_fn(out_channels)

#         self.dLdW = np.zeros(self.W.shape)
#         self.dLdb = np.zeros(self.b.shape)

#     def forward(self, A):
#         """
#         Argument:
#             A (np.array): (batch_size, in_channels, input_height, input_width)
#         Return:
#             Z (np.array): (batch_size, out_channels, output_height, output_width)
#         """
        
#         self.A = A

#         # batch_size, in_channels, input_height, input_width = A.shape
#         kernel_size = self.kernel_size

#         # Extract patches
#         A_patches = np.lib.stride_tricks.sliding_window_view(
#             A, (kernel_size, kernel_size), axis=(2, 3))
#         # A_patches shape: (batch_size, in_channels, output_height, output_width, kernel_size, kernel_size)

#         # Rearrange axes to (batch_size, output_height, output_width, in_channels, kernel_size, kernel_size)
#         A_patches = A_patches.transpose(0, 2, 3, 1, 4, 5)
#         self.A_patches = A_patches  # Store for backward pass

#         # Compute Z using tensordot
#         Z = np.tensordot(A_patches, self.W, axes=([3, 4, 5], [1, 2, 3]))
#         # Z shape: (batch_size, output_height, output_width, out_channels)

#         # Rearrange axes to (batch_size, out_channels, output_height, output_width)
#         Z = Z.transpose(0, 3, 1, 2)

#         # Add bias
#         Z += self.b[np.newaxis, :, np.newaxis, np.newaxis]

#         return Z

#     def backward(self, dLdZ):
#         """
#         Argument:
#             dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
#         Return:
#             dLdA (np.array): (batch_size, in_channels, input_height, input_width)
#         """
        
#         # Gradient w.r.t. bias
#         self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

#         # Gradient w.r.t. weights
#         self.dLdW = np.tensordot(dLdZ, self.A_patches, axes=([0, 2, 3], [0, 1, 2]))
#         # Resulting shape: (out_channels, in_channels, kernel_size, kernel_size)

#         # Flip W and swap axes
#         W_flipped = self.W[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)  # Shape: (in_channels, out_channels, kernel_size, kernel_size)

#         # Pad dLdZ
#         pad_h = self.kernel_size - 1
#         pad_w = self.kernel_size - 1

#         dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

#         # Extract patches from dLdZ_padded
#         dLdZ_patches = np.lib.stride_tricks.sliding_window_view(
#             dLdZ_padded, (self.kernel_size, self.kernel_size), axis=(2, 3))
#         # Shape: (batch_size, out_channels, input_height, input_width, kernel_size, kernel_size)

#         # Rearrange axes to (batch_size, input_height, input_width, out_channels, kernel_size, kernel_size)
#         dLdZ_patches = dLdZ_patches.transpose(0, 2, 3, 1, 4, 5)

#         # Compute dLdA using tensordot
#         dLdA = np.tensordot(dLdZ_patches, W_flipped, axes=([3, 4, 5], [1, 2, 3]))
#         # dLdA shape: (batch_size, input_height, input_width, in_channels)

#         # Rearrange axes to (batch_size, in_channels, input_height, input_width)
#         dLdA = dLdA.transpose(0, 3, 1, 2)

#         return dLdA


# class Conv2d():
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
#                  weight_init_fn=None, bias_init_fn=None):
#         # Do not modify the variable names
#         self.stride = stride
#         self.pad = padding

#         # Initialize Conv2d() and Downsample2d() isntance
#         self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
#         self.downsample2d = Downsample2d(stride)  # TODO

#     def forward(self, A):
#         """
#         Argument:
#             A (np.array): (batch_size, in_channels, input_height, input_width)
#         Return:
#             Z (np.array): (batch_size, out_channels, output_height, output_width)
#         """
        
#         # Pad the input appropriately using np.pad() function
#         A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant') # TODO

#         # Call Conv2d_stride1
#         Z = self.conv2d_stride1.forward(A_padded) # TODO

#         # downsample
#         Z = self.downsample2d.forward(Z)  # TODO

#         return Z

#     def backward(self, dLdZ):
#         """
#         Argument:
#             dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
#         Return:
#             dLdA (np.array): (batch_size, in_channels, input_height, input_width)
#         """

#         # Call downsample1d backward
#         dLdZ_upsampled = self.downsample2d.backward(dLdZ) # TODO

#         # Call Conv1d_stride1 backward
#         dLdA_padded = self.conv2d_stride1.backward(dLdZ_upsampled)  # TODO

#         # Unpad the gradient
#         # TODO
#         if self.pad > 0:
#             dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
#         else:
#             dLdA = dLdA_padded

#         return dLdA
    
    
    
class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, _, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        # Extract patches using stride tricks
        self.A_patches = np.lib.stride_tricks.sliding_window_view(
            A, (self.kernel_size, self.kernel_size), axis=(2, 3)
        ).transpose(0, 2, 3, 1, 4, 5)

        # Compute convolution using tensordot
        Z = np.tensordot(self.A_patches, self.W, axes=([3, 4, 5], [1, 2, 3]))

        # Rearrange axes and add bias
        Z = Z.transpose(0, 3, 1, 2) + self.b[np.newaxis, :, np.newaxis, np.newaxis]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Gradient w.r.t. bias
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        # Gradient w.r.t. weights
        self.dLdW = np.tensordot(dLdZ, self.A_patches, axes=([0, 2, 3], [0, 1, 2]))

        # Prepare for backward pass
        W_flipped = np.flip(self.W, axis=(2, 3)).transpose(1, 0, 2, 3)
        
        pad_h = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (pad_h, pad_h), (pad_h, pad_h)), mode='constant')

        # Extract patches from padded dLdZ
        dLdZ_patches = np.lib.stride_tricks.sliding_window_view(
            dLdZ_padded, (self.kernel_size, self.kernel_size), axis=(2, 3)
        ).transpose(0, 2, 3, 1, 4, 5)

        # Compute dLdA using tensordot
        dLdA = np.tensordot(dLdZ_patches, W_flipped, axes=([3, 4, 5], [1, 2, 3]))

        return dLdA.transpose(0, 3, 1, 2)


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        Z = self.conv2d_stride1.forward(A_padded)
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_upsampled)
        
        if self.pad > 0:
            return dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return dLdA_padded