import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # 
        self.b = np.zeros((out_features, 1))  # 

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  
        self.N = np.shape(A)[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z =( self.A @ self.W.T) + (self.Ones * self.b.T)  # TODO
        
        # print("Shape of A: ", np.shape(self.A))
        # print(f"Shape of W: {np.shape(self.W)}")
        # print(f"Shape of b: {np.shape(self.b)}")
        # print(f"Shape of Ones: {np.shape(self.Ones)}")
        # print("Shape of N: ", np.shape(self.N))
        # print(f"Shape of Z: {np.shape(Z)}")
        
        # print("A: ", self.A)
        # print("W: ", self.W)
        # print("b: ", self.b)
        # print("Ones: ", self.Ones)
        # print("N: ", self.N)
    

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ @ self.W  # TODO
        self.dLdW = dLdZ.T @ self.A  # TODO
        self.dLdb = dLdZ.T @ self.Ones  # TODO
        
        # # update bias and weights
        # self.W = self.W - self.dLdW
        # self.b = self.b - self.dLdb

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
