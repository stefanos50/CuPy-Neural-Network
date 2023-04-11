import cupy as cp

def mean_squared_error(actual, predicted):
    diff = predicted - actual
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    return mean_diff

def cross_entropy_loss(actual, predicted,eps=1e-7):
    predictions = cp.clip(predicted, eps, 1. - eps)
    N = predictions.shape[0]
    loss = -cp.sum(actual*cp.log(predictions+eps))/N
    return loss

def calculate_loss(selected_loss,actual,predicted):
    if selected_loss=="mse":
        return mean_squared_error(actual,predicted)
    elif selected_loss=="cross-entropy":
        return cross_entropy_loss(actual,predicted)