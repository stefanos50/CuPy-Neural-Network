import cupy as cp

def GlorotNormal(input_size,output_size):
    limit = cp.sqrt(2.0 / (input_size + output_size))
    return cp.random.normal(0.0,limit,size=(input_size, output_size))

def GlorotUniform(input_size,output_size):
    limit = cp.sqrt(6.0 / (input_size + output_size))
    return cp.random.uniform(-limit,limit,size=(input_size, output_size))

def RandomNormal(min=0,max=1,input_size=0,output_size=0):
    return cp.random.normal(min,max,size=(input_size, output_size))

def RandomUniform(min=0,max=1,input_size=0,output_size=0):
    return cp.random.uniform(min,max,size=(input_size, output_size))