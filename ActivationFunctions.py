import cupy as cp

def relu(x,prime=False):
    if prime:
       return cp.where(x <= 0, 0, 1)
    return cp.maximum(0,x)

def softmax(x,prime=False):
    exp = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    if prime:
        return exp / cp.sum(exp, axis=0) * (1 - exp / cp.sum(exp, axis=0))
    return exp / cp.sum(exp, axis=1, keepdims=True)

def sigmoid(x,prime=False):
    if prime:
        sigmoid_basic = 1 / (1 + cp.exp(-x))
        return sigmoid_basic * (1 - sigmoid_basic)
    return 1 / (1 + cp.exp(-x))