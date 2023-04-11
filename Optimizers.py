import cupy as cp

def vanilla_gradient_decsent(layer_object,wight,bias,h):
    layer_object.Weights = cp.subtract(layer_object.Weights ,h * wight)
    layer_object.bias -= cp.subtract(layer_object.bias , h * bias)

def adam(learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,current_layer_index=0,network_layers=None,weight=None,bias=None,step=0,m=None,v=None,params=["Weight","Bias"]):

    grads = [weight,bias]

    for i in range(len(params)):
        m["layer_" + str(current_layer_index) + "_"+str(params[i])] = beta1 * m["layer_" + str(current_layer_index) + "_"+str(params[i])] + (1-beta1) * grads[i]
        v["layer_"+str(current_layer_index)+ "_"+str(params[i])] = beta2 * v["layer_" + str(current_layer_index) + "_"+str(params[i])] + (1-beta2) * grads[i] ** 2

        m_corrected = m["layer_" + str(current_layer_index) + "_"+str(params[i])] / (1 - beta1 ** step)
        v_corrected = v["layer_"+str(current_layer_index)+ "_"+str(params[i])] / (1 - beta2 ** step)

        updated = - learning_rate * m_corrected / (cp.sqrt(v_corrected) + eps)

        if params[i] == "Weight":
            network_layers[current_layer_index].Weights = cp.add(network_layers[current_layer_index].Weights,updated)
        elif params[i] == "Bias":
            network_layers[current_layer_index].bias = cp.add(network_layers[current_layer_index].bias,updated)

    return m,v

def ADAGrad(learning_rate=1e-3,eps=1e-8,grad_weight=None,grad_bias=None,network_layers=None,current_layer_index=0,c=None,params=["Weight","Bias"]):

    grads = [grad_weight, grad_bias]

    for i in range(len(params)):
        c["layer_"+str(current_layer_index)+ "_"+str(params[i])] =  c["layer_"+str(current_layer_index)+ "_"+str(params[i])] + grads[i] ** 2
        updated = - learning_rate * grads[i] / (cp.sqrt(c["layer_"+str(current_layer_index)+ "_"+str(params[i])])+eps)

        if params[i] == "Weight":
            network_layers[current_layer_index].Weights = cp.add(network_layers[current_layer_index].Weights,updated)
        elif params[i] == "Bias":
            network_layers[current_layer_index].bias = cp.add(network_layers[current_layer_index].bias,updated)

    return c

def RMSprop(learning_rate=1e-3,decay_rate=0.9,eps=1e-8,grad_weight=None,grad_bias=None,network_layers=None,current_layer_index=0,c=None,params=["Weight","Bias"]):
    grads = [grad_weight, grad_bias]

    for i in range(len(params)):
        c["layer_"+str(current_layer_index)+ "_"+str(params[i])] = decay_rate * c["layer_"+str(current_layer_index)+ "_"+str(params[i])] + (1-decay_rate) * grads[i] ** 2
        updated = - learning_rate * grads[i] / (cp.sqrt(c["layer_"+str(current_layer_index)+ "_"+str(params[i])])+eps)

        if params[i] == "Weight":
            network_layers[current_layer_index].Weights = cp.add(network_layers[current_layer_index].Weights,updated)
        elif params[i] == "Bias":
            network_layers[current_layer_index].bias = cp.add(network_layers[current_layer_index].bias,updated)

    return c

def SGDM(learning_rate=1e-3,momentum=0.2,grad_weight=None,grad_bias=None,network_layers=None,current_layer_index=0,c=None,params=["Weight","Bias"]):
    grads = [grad_weight, grad_bias]

    for i in range(len(params)):
        c["layer_"+str(current_layer_index)+ "_"+str(params[i])] = - learning_rate *  grads[i] +  c["layer_"+str(current_layer_index)+ "_"+str(params[i])] * momentum

        if params[i] == "Weight":
            network_layers[current_layer_index].Weights = cp.add(network_layers[current_layer_index].Weights,c["layer_"+str(current_layer_index)+ "_"+str(params[i])])
        elif params[i] == "Bias":
            network_layers[current_layer_index].bias = cp.add(network_layers[current_layer_index].bias,c["layer_"+str(current_layer_index)+ "_"+str(params[i])])

    return c