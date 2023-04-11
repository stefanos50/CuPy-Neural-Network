import cupy as cp
from sklearn.model_selection import train_test_split
import ActivationFunctions
import LossFunctions
import Optimizers
import WeightInitializers

class Network_Layer:
    def __init__(self,hidden_neurons:int=2,activation_function:str="ReLu",weight_initializer="glotor_normal",init_min=0,init_max=1):
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function
        self.Weights = cp.array([])
        self.bias = None
        self.input = None
        self.weight_initializer = weight_initializer
        self.init_min = init_min
        self.init_max = init_max
        self.derivative_weights = None
        self.derivative_bias = None
        self.derivative_result = None


    def initialize_network(self,F_in,F_out):
        self.bias = cp.zeros((1, F_out))

        if self.weight_initializer == "glotor_uniform":
            self.Weights = WeightInitializers.GlorotUniform(F_in,F_out)
        elif self.weight_initializer == "glotor_normal":
            self.Weights = WeightInitializers.GlorotNormal(F_in, F_out)
        elif self.weight_initializer == "random_normal":
            self.Weights = WeightInitializers.RandomNormal(self.init_min,self.init_max,F_in, F_out)
        elif self.weight_initializer == "uniform_normal":
            self.Weights = WeightInitializers.RandomUniform(self.init_min,self.init_max,F_in, F_out)
        else:
            raise Exception("No weight_initializer defined. Available weight_initializers -> glotor_uniform,glotor_normal,random_normal,random_uniform.")


    def get_activation_function(self,x,prime=False):
        if self.activation_function == "ReLu":
            return ActivationFunctions.relu(x,prime)
        elif self.activation_function == "sigmoid":
            return ActivationFunctions.sigmoid(x,prime)
        elif self.activation_function == "softmax":
            return ActivationFunctions.softmax(x,prime)

    def forward_pass(self,X):
        self.input = cp.array(X)

        if cp.any(self.Weights) == False:
            self.initialize_network(self.input.shape[1],self.hidden_neurons)

        self.Result = cp.matmul(X,self.Weights) + self.bias
        if self.activation_function == None:
            raise Exception("No activation function defined. Available activation functions -> ReLu,softmax,sigmoid.")
        self.Activate = self.get_activation_function(self.Result,False)

        return self.Result

    def backward(self,next_layer,num_labels):
        self.derivative_result = cp.matmul(
            next_layer.derivative_result,
            next_layer.Weights.T) * self.get_activation_function(self.Result, True)
        self.derivative_weights = 1 / num_labels * cp.matmul(
            self.input.T, self.derivative_result)
        self.derivative_bias = 1 / num_labels * cp.sum(
            self.derivative_result, axis=0, keepdims=True)

class NNet:
    def __init__(self):
        self.layers_tracking_dict = {}
        self.backpropagation_info = {}
        self.optimizer_cache_f = {}
        self.optimizer_cache_s = {}
        self.history = {}
        self.verbose_levels = [0,1,10,100,1000]



    def add_layer(self,new_layer):
        try:
            self.layers_tracking_dict[max(self.layers_tracking_dict.keys()) + 1] = new_layer
        except:
            self.layers_tracking_dict[0 + 1] = new_layer


    def NNet_Configuration(self,epochs,learning_rate,optimizer=None,verbose=0,momentum=0.2,validation_split=0.33):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.verbose = verbose
        self.momentum = momentum
        self.validation_split = validation_split

        if self.optimizer == "adam":
            for id in self.layers_tracking_dict.keys():
                self.optimizer_cache_f["layer_" + str(id) + "_Weight"] = 0
                self.optimizer_cache_f["layer_" + str(id) + "_Bias"] = 0
                self.optimizer_cache_s["layer_" + str(id) + "_Weight"] = 0
                self.optimizer_cache_s["layer_" + str(id) + "_Bias"] = 0
        else:
            self.optimizer_cache_s.clear()
            for id in self.layers_tracking_dict.keys():
                self.optimizer_cache_f["layer_" + str(id) + "_Weight"] = 0
                self.optimizer_cache_f["layer_" + str(id) + "_Bias"] = 0

    def forward(self,x):
        for layer_id, layer_object in self.layers_tracking_dict.items():
            x = layer_object.forward_pass(x)
        return x

    def backwards_pass(self,y_true):
        last_layer_index = max(self.layers_tracking_dict.keys()) #get the last (output layer) index
        num_labels = y_true.shape[0]

        for layer_index in reversed(range(1,last_layer_index+1)):
            if layer_index == last_layer_index:
                self.layers_tracking_dict[layer_index].derivative_result = self.layers_tracking_dict[layer_index].Activate - y_true
                self.layers_tracking_dict[layer_index].derivative_weights = 1 / num_labels * cp.matmul(self.layers_tracking_dict[layer_index].input.T,self.layers_tracking_dict[layer_index].derivative_result)
                self.layers_tracking_dict[layer_index].derivative_bias = 1 / num_labels * cp.sum(self.layers_tracking_dict[layer_index].derivative_result, axis=0, keepdims=True)
                continue
            self.layers_tracking_dict[layer_index].backward(self.layers_tracking_dict[layer_index+1],num_labels)

    def split_into_batches(self,data, labels, batch_size):
        number_of_elements = data.shape[0]
        number_of_batches = int(number_of_elements / batch_size + 1)
        batches = []
        for i in range(number_of_batches):
            batches.append((data[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size]))

        if number_of_elements % batch_size == 0:
            batches.pop(-1)
        return batches

    def predict(self,x=None,y=None):
        network_predictions = self.forward(x)
        return list(cp.argmax(network_predictions, axis=1) == cp.argmax(y, axis=1)).count(True) / len(cp.argmax(network_predictions, axis=1) == cp.argmax(y, axis=1))

    def print_progress(self,phase,accuracy,loss,current_epoch):
        if current_epoch%self.verbose_levels[self.verbose] ==0:
            print("Phase "+str(phase)+" - Accuracy: "+str(accuracy*100)+"%"+" - "+"loss: "+str(loss))

    def evaluate(self,x=None,y=None,batch_size=32):
        accuracy_cache = []
        data_bached = self.split_into_batches(x, y, batch_size)
        for x_data, y_labels in data_bached:
            accuracy_cache.append(self.predict(x_data,y_labels))
        return sum(accuracy_cache) / len(accuracy_cache)


    def fit_NNet(self,X_train,y_train,batch_size=32,val_batch_size=None,loss_function="mse"):
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=self.validation_split,random_state=42)
        if val_batch_size == None:
            val_batch_size = batch_size
        total_train_loss_tracker = []
        total_val_loss_tracker = []
        total_train_accuracy_tracker = []
        total_val_accuracy_tracker = []
        for i in range(1,self.epochs+1):
            for phase in ['train', 'val']:
                epoch_loss_tracker = []
                epoch_accuracy_tracker = []
                if phase == 'train':
                    step = 0
                    if (self.verbose != 0):
                        if i % self.verbose_levels[self.verbose] == 0:
                            print("\n")
                            print("Epoch: "+str(i)+"/"+str(self.epochs)+" - ║{0:20s}║ {1:.1f}%".format('🟩' * int(i/self.epochs * 20), i/self.epochs * 100))
                    data_bached = self.split_into_batches(X_train,y_train,batch_size)
                    for x_data,y_labels in data_bached:
                        step = step + 1
                        network_predictions = self.forward(x_data)
                        calculated_loss = LossFunctions.calculate_loss(loss_function,y_labels,network_predictions)
                        calculated_accuracy = self.predict(x_data,y_labels)
                        epoch_loss_tracker.append(calculated_loss)
                        epoch_accuracy_tracker.append(calculated_accuracy)

                        self.backwards_pass(y_labels)


                        for layer_index in range(1, max(self.layers_tracking_dict.keys()) + 1):
                            if self.optimizer == "VGD":
                                Optimizers.vanilla_gradient_decsent(self.layers_tracking_dict[layer_index],self.layers_tracking_dict[layer_index].derivative_weights,self.layers_tracking_dict[layer_index].derivative_bias,self.learning_rate)
                            elif self.optimizer == "adam":
                                self.optimizer_cache_f , self.optimizer_cache_s = Optimizers.adam(learning_rate=self.learning_rate,current_layer_index=layer_index,network_layers=self.layers_tracking_dict,weight=self.layers_tracking_dict[layer_index].derivative_weights,bias=self.layers_tracking_dict[layer_index].derivative_bias,step=step,m=self.optimizer_cache_f,v=self.optimizer_cache_s)
                            elif self.optimizer == "RMSprop":
                                self.optimizer_cache_f = Optimizers.RMSprop(learning_rate=self.learning_rate,grad_weight=self.layers_tracking_dict[layer_index].derivative_weights,grad_bias=self.layers_tracking_dict[layer_index].derivative_bias,network_layers=self.layers_tracking_dict,current_layer_index=layer_index,c=self.optimizer_cache_f)
                            elif self.optimizer == "ADAGrad":
                               self.optimizer_cache_f = Optimizers.ADAGrad(learning_rate=self.learning_rate,grad_weight=self.layers_tracking_dict[layer_index].derivative_weights,grad_bias=self.layers_tracking_dict[layer_index].derivative_bias,network_layers=self.layers_tracking_dict,current_layer_index=layer_index,c=self.optimizer_cache_f)
                            elif self.optimizer == "SGDM":
                                self.optimizer_cache_f = Optimizers.SGDM(learning_rate=self.learning_rate,momentum=self.momentum,grad_weight=self.layers_tracking_dict[layer_index].derivative_weights,grad_bias=self.layers_tracking_dict[layer_index].derivative_bias,network_layers=self.layers_tracking_dict,current_layer_index=layer_index,c=self.optimizer_cache_f)


                    total_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
                    total_train_loss_tracker.append(float(total_loss))

                    total_accuracy = sum(epoch_accuracy_tracker) / len(epoch_accuracy_tracker)
                    total_train_accuracy_tracker.append(float(total_accuracy))

                    if(self.verbose != 0):
                        self.print_progress(phase,total_train_accuracy_tracker[-1],total_loss,i)
                elif phase == 'val':
                    data_bached = self.split_into_batches(X_validation, y_validation, val_batch_size)
                    for x_data, y_labels in data_bached:
                        network_predictions = self.forward(x_data)
                        calculated_loss = LossFunctions.calculate_loss(loss_function, y_labels, network_predictions)
                        calculated_accuracy = self.predict(x_data,y_labels)
                        epoch_loss_tracker.append(calculated_loss)
                        epoch_accuracy_tracker.append(calculated_accuracy)

                    total_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
                    total_val_loss_tracker.append(float(total_loss))

                    total_accuracy = sum(epoch_accuracy_tracker) / len(epoch_accuracy_tracker)
                    total_val_accuracy_tracker.append(float(total_accuracy))
                    #validation_set_accuracy.append(self.predict(X_validation,y_validation))
                    if(self.verbose != 0):
                        self.print_progress(phase,total_val_accuracy_tracker[-1],total_loss,i)

        self.history["loss"] = total_train_loss_tracker
        self.history["val_loss"] = total_val_loss_tracker
        self.history["accuracy"] = total_train_accuracy_tracker
        self.history["val_accuracy"] = total_val_accuracy_tracker


