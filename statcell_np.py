import numpy as np

class Cell:

    def __init__(self, inputs, size=32):
        self.inputs = inputs
        self.size = size
        self.c_acc = np.zeros(self.inputs)
        self.s_acc = np.zeros(self.inputs)
        self.avg = np.zeros(self.inputs)
        self.var = np.zeros(self.inputs)
        self.weights = np.random.random((self.inputs, size))
        self.is_training = True
    
    def train(self):
        self.is_training = True
    
    def eval(self):
        self.is_training = False
    
    def step(self, data):
        self.s_acc += data
        self.c_acc += 1
        self.avg = self.s_acc / self.c_acc
        self.var += (data - self.avg)**2
        data = np.round((data - self.avg) / np.sqrt(self.var) * self.size * 0.5)
        