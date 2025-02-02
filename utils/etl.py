from keras.datasets import mnist
from sklearn.datasets import load_digits

def etl(num_train, num_test, model_types):

    # Collect data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # setup data
    train_X = train_X[:num_train].astype(int)
    train_y = train_y[:num_train].astype(int)
    test_X = test_X[:num_test].astype(int)
    test_y = test_y[:num_test].astype(int)

    # setup output
    results = {}
    for model in model_types:
        results[model] = {}

    return results, (train_X, train_y, test_X, test_y)

def etl_test(num_train, num_test, model_types):
    
    # Collect data
    digits = load_digits()
    
    # setup data
    X = digits.data
    y = digits.target
    
    train_X = X[:num_train].astype(int)
    train_y = y[:num_train].astype(int)
    test_X = X[num_train:(num_train + num_test)].astype(int)
    test_y = y[num_train:(num_train + num_test)].astype(int)
    
    # setup output
    results = {}
    for model in model_types:
        results[model] = {}
    
    return results, (train_X, train_y, test_X, test_y)
