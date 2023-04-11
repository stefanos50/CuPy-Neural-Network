import time
from sklearn.model_selection import train_test_split
import DataPreprocessing
import cupy as cp
import NNet
import HelperMethods
import KerasNNet

def run(X_train,X_test,y_train,y_test,keras_model):

    model = NNet.NNet()
    model.add_layer(
        NNet.Network_Layer(32, activation_function='ReLu', weight_initializer="glotor_uniform", init_min=0, init_max=1))
    # model.add_layer(Network_Layer(5, activation_function='ReLu'))
    model.add_layer(
        NNet.Network_Layer(10, activation_function='softmax', weight_initializer="glotor_uniform", init_min=0,
                           init_max=1))
    model.NNet_Configuration(epochs=150, learning_rate=0.001, optimizer="adam", verbose=1, validation_split=0.33)

    start = time.time()
    model.fit_NNet(cp.array(X_train.tolist()), cp.array(y_train.tolist()), batch_size=32, loss_function="cross-entropy")
    end = time.time()


    accuracy_train = round(model.evaluate(cp.array(X_train), cp.array(y_train), 32) * 100, 2)
    accuracy_test = round(model.evaluate(cp.array(X_test), cp.array(y_test),32) * 100, 2)


    if keras_model:
        keras_history,keras_time,keras_train_accuracy,keras_test_accuracy = KerasNNet.run_keras_model(X_train, X_test, y_train, y_test)

    if keras_model:
        return model.history,end - start,accuracy_train,accuracy_test,keras_history,keras_time,keras_train_accuracy,keras_test_accuracy
    else:
        return model.history, end - start, accuracy_train, accuracy_test