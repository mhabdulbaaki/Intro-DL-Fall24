import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        Win = A.shape[2]
        indeces = list(range(1, Win))
        expand = np.repeat(indeces, self.upsampling_factor-1)
       
        Z = np.insert(A, expand, 0, axis=2)  # TODO

        # b, c, i = A.shape
        # o = self.upsampling_factor * (i -1) +1
        # Z = np.zeros((b , c , o))
        # Z[:, :, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor]  # TODO

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        Z = A[:, :, ::self.downsampling_factor]  # TODO
        
        self.Win = A.shape[2]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        # indeces = list(range(1, self.Win))
        # expand = np.repeat(indeces, self.downsampling_factor)       

        # dLdA = np.insert(dLdZ, expand, 0, axis=1)  # TODO 
        
        r,c,_ = dLdZ.shape
        dLdA = np.zeros((r, c, self.Win))
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA



class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        
        
        b, c, Hin,  Win = A.shape
        shape = (b, c, (self.upsampling_factor * (Hin - 1) + 1), (self.upsampling_factor * (Win - 1) + 1 ))
        Z = np.zeros(shape) # TODO
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]  # TODO

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        Z = A[:,:,::self.downsampling_factor,::self.downsampling_factor]   # TODO
        _, __, self.Hin,  self.Win = A.shape

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        
        b, c, _,  __ = dLdZ.shape
        shape = (b, c, self.Hin, self.Win)
        dLdA = np.zeros(shape) # TODO
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

    

        return dLdA
