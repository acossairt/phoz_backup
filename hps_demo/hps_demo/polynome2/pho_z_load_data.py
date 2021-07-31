import pickle

def load_data():
    X_train, y_train = pickle.load(open( b"/data/a/cpac/aurora/MDN_phoZ/training_data.obj", "rb") )
    X_test, y_test = pickle.load(open( b"/data/a/cpac/aurora/MDN_phoZ/validation_data.obj", "rb") )
    return X_train, y_train, X_test, y_test