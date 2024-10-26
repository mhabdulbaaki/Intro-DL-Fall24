import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
    
        res = 1 / (1 + np.exp(-Z))
        
        self.A = res
        return res
    
    def backward(self, dLdA):
       return dLdA * (self.A * (1 - self.A))


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        
        
        numerator = np.exp(Z) - np.exp(-Z)
        denuminator = np.exp(Z) + np.exp(-Z)
        # res = numerator / denuminator
        results = np.tanh(Z)
        self.A = results
        return results
    
    def backward(self, dLdA):
        return dLdA * (1 - self.A**2)

class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        res = np.maximum(0, Z)
        self.A = res
        return res
    
    def backward(self, dLdA):
        # if self.A > 0:
        #     return dLdA * 1
        # else :
        #     return dLdA * 0
        res = dLdA * (self.A > 0)
        return res

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):
        erfZ = scipy.special.erf(Z/np.sqrt(2))
        res = (Z/2) * (1 + erfZ)
        self.A = Z
        return res
    
    def backward(self, dLdA):
        erfZ = scipy.special.erf(self.A/np.sqrt(2))
        res = (1/2 * (1 + erfZ)) + (self.A / np.sqrt(2 * np.pi)) * np.exp(-self.A**2 / 2)
        return dLdA * res
     

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """

       
        n = np.exp(Z) 
        d = np.sum(np.exp(Z), axis=1, keepdims=True)
        
        self.A = n / d
        
        print("Matrix Z: ", Z)
        
        return n / d
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = np.shape(dLdA)[0] # TODO
        C = np.shape(dLdA)[1] # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N,C)) # TODO

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C,C)) # TODO

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m,n] = self.A[i,m] * (1-self.A[i,m]) # TODO
                    else:
                        J[m,n] = -self.A[i,m] * (self.A[i,n]) # TODO

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i] @ J # TODO

        return dLdZ