import time
from keras import backend as K, Sequential
from keras.layers import Dense


def run_keras_model(X_train,X_test,y_train,y_test):
    model = Sequential()
    model.add(Dense(32, input_shape=(64,), activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile the keras model
    model.compile(loss='CategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])
    K.set_value(model.optimizer.learning_rate, 0.001)
    # fit the keras model on the dataset
    start = time.time()
    history = model.fit(X_train, y_train, epochs=150, batch_size=32,validation_split=0.33)
    end = time.time()

    _, accuracy_train = model.evaluate(X_train, y_train)
    _, accuracy_test = model.evaluate(X_test, y_test)


    return history.history,end - start,round(accuracy_train*100,2),round(accuracy_test*100,2)
