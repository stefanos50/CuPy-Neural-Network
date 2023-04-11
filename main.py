from sklearn.model_selection import train_test_split, KFold
import DataPreprocessing
import HelperMethods
import Run
cross_validation = True
K = 2
keras_model = True

x, y = DataPreprocessing.get_digits_dataset()

def plot_results(history,train_accuracy,test_accuracy,time,type):
    HelperMethods.plot_result(history['loss'], history['val_loss'], "Loss Plot "+str(type), "Epochs", "Loss",
                              "Train Loss", "Val Loss")
    HelperMethods.plot_result(history['accuracy'], history['val_accuracy'], "Accuracy Plot "+str(type), "Epochs",
                              "Accuracy", "Train Accuracy", "Val Accuracy")


    print("\n")
    print(str(type)+" train accuracy: " + str(train_accuracy) + "%")
    print(str(type)+" test accuracy: " + str(test_accuracy) + "%")
    print(str(type)+" fit execution time: " + str(time) + "s")

def history_average(history):
    loss = []
    val_loss = []
    accuracy = []
    val_accuracy = []
    for dict in history:
        loss.append(dict['loss'])
        val_loss.append(dict['val_loss'])
        accuracy.append(dict['accuracy'])
        val_accuracy.append(dict['val_accuracy'])

    history_new = {}
    history_new['loss'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*loss)]
    history_new['val_loss'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*val_loss)]
    history_new['accuracy'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*accuracy)]
    history_new['val_accuracy'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*val_accuracy)]
    return history_new

if cross_validation == False:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    if keras_model is True:
        history,time,train_accuracy,test_accuracy,keras_history,keras_time,keras_train_accuracy,keras_test_accuracy = Run.run(X_train,X_test,y_train,y_test,keras_model)
    else:
        history, time, train_accuracy, test_accuracy = Run.run(X_train, X_test, y_train, y_test, keras_model)
    plot_results(history,train_accuracy,test_accuracy,time,"CuPy")
    try:
        plot_results(keras_history, keras_train_accuracy, keras_test_accuracy, keras_time, "Keras")
    except:
        pass
else:
    kf = KFold(n_splits=K)
    historyls = []
    keras_historyls = []
    timels = []
    keras_timels = []
    test_accuracyls = []
    train_accuracyls = []
    keras_test_accuracyls = []
    keras_train_accuracyls = []
    for train_index, test_index in kf.split(x):
        train_data_x, test_data_x = x[train_index], x[test_index]
        train_data_y, test_data_y = y[train_index], y[test_index]

        if keras_model is True:
            history, time, train_accuracy, test_accuracy, keras_history, keras_time, keras_train_accuracy, keras_test_accuracy = Run.run(train_data_x, test_data_x, train_data_y, test_data_y, keras_model)
            historyls.append(history)
            timels.append(time)
            test_accuracyls.append(test_accuracy)
            train_accuracyls.append(train_accuracy)

            keras_historyls.append(keras_history)
            keras_timels.append(keras_time)
            keras_train_accuracyls.append(keras_train_accuracy)
            keras_test_accuracyls.append(keras_test_accuracy)
        else:
            history, time, train_accuracy, test_accuracy = Run.run(train_data_x, train_data_y, test_data_x, test_data_y,keras_model)
            historyls.append(history)
            timels.append(time)
            test_accuracyls.append(test_accuracy)
            train_accuracyls.append(train_accuracy)


    plot_results(history_average(historyls), sum(train_accuracyls)/len(train_accuracyls), sum(test_accuracyls)/len(test_accuracyls), sum(timels)/len(timels), "Cross Validation Avg CuPy")
    try:
        plot_results(history_average(keras_historyls), sum(keras_train_accuracyls)/len(keras_train_accuracyls), sum(keras_test_accuracyls)/len(keras_test_accuracyls), sum(keras_timels)/len(keras_timels), "Cross Validation Avg Keras")
    except:
        pass




