import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = np.shape(self.Z)[0]   # TODO
        self.M = np.mean(self.Z, axis=0)   # TODO
        self.V = np.var(self.Z, axis=0)  # TODO
        
        
        z_sub_mean = self.Z - self.M
        v_add_eps = self.V + self.eps 
        
        
        if eval == False:
            # training mode
            self.NZ = (z_sub_mean) / np.sqrt(v_add_eps)  # TODO
            self.BZ = self.BW * self.NZ + self.Bb   # TODO

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M  # TODO
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V  # TODO
            
            return self.BZ
        else:
            # inference mode
            NZ = NZ = (self.Z - self.running_M) / np.sqrt(self.running_V + self.eps)  # TODO
            BZ = self.BW * NZ + self.Bb   # TODO

        return BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0)  # TODO
        self.dLdBb = np.sum(dLdBZ, axis=0)  # TODO

        dLdNZ = dLdBZ * self.BW  # TODO
        
        v_add_eps = self.V + self.eps
        z_sub_mean = self.Z - self.M
        
        dLdV = np.sum(dLdNZ * z_sub_mean * -0.5 * v_add_eps**(-3/2), axis=0)  # TODO
        dLdM = np.sum(dLdNZ * -1 / np.sqrt(v_add_eps), axis=0) + dLdV * np.sum(-2 * (z_sub_mean), axis=0) / self.N  # TODO

        dLdZ = dLdNZ / np.sqrt(v_add_eps) + dLdV * 2 * (z_sub_mean) / self.N + dLdM / self.N  # TODO

        return dLdZ
