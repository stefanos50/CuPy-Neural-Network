import sklearn as sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
import pandas as pd

def convert_to_one_hot_encode(labels):
    return LabelBinarizer().fit_transform(labels)

def get_digits_dataset():
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data
    y = digits.target
    y = convert_to_one_hot_encode(y)
    return X,y


get_digits_dataset()