# -*- coding: utf-8 -*-
import numpy as np

class squaredError:
    def loss(self, Y, predY):
        return np.sum(np.square(Y - predY))

    def backward(self, Y, predY):
        return 2*(Y-predY)

