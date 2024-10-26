import numpy as np

class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = np.shape(A)[0]  # TODO
        self.C = np.shape(A)[1]  # TODO
        se = (self.A - self.Y) * (self.A - self.Y)  # TODO
        c = np.ones(self.C)
        n = np.ones(self.N)
        sse = n.T @ se @ c  # TODO
        mse = sse / (self.N * self.C)  # TODO

        return mse

    def backward(self):

        dLdA = (2 * (self.A - self.Y)) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = np.shape(self.A)[0]  # TODO
        C = np.shape(self.Y)[1]  # TODO

        Ones_C = np.ones(C)  # TODO
        Ones_N = np.ones(N)  # TODO

        n = np.exp(self.A) 
        d = np.sum(np.exp(self.A), axis=1, keepdims=True)
        
        self.softmax = n / d

        # self.softmax = activation.Softmax()  # TODO
        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C  # TODO
        sum_crossentropy = Ones_N @ crossentropy  # TODO
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / np.shape(self.A)[0] # TODO

        return dLdA
