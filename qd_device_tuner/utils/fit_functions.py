import numpy as np

def logarithmic(x, a, b, x0, y0):
    return a * np.log(b*(x-x0)) + y0

def exponential(x, a, b, x0, y0):
    return a * np.exp(b * (x-x0)) + y0

def sigmoid(x, a, b, x0, y0):
    return a/(1+np.exp(b * (x-x0))) + y0

def linear(x, m, b):
    return m * x + b         