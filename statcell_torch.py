import torch

class Cell:
    def __init__(self, inputs, size=32):
        self.inputs = inputs
        self.size = size
        self.c_acc = 1
        self.s_acc = torch.ones(self.inputs)
        self.avg = torch.zeros(self.inputs)
        self.var = torch.zeros(self.inputs)
        self.weights = torch.rand(self.inputs, size)
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
        data = torch.round((data - self.avg) /
               torch.sqrt(self.var + 0.001) * self.size * 0.5)
        return data

