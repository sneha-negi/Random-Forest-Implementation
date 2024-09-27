import numpy as np
from data.data_processing_titanic import X_train as X,y_train as y

def draw_bootstrap(X_train, y_train):
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
    oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
    X_bootstrap = X_train[bootstrap_indices]
    y_bootstrap = y_train[bootstrap_indices]
    X_oob = X_train[oob_indices]
    y_oob = y_train[oob_indices]
    return X_bootstrap, y_bootstrap, X_oob, y_oob

if __name__ == '__main__':
    temp_train = draw_bootstrap(X,y)
    print(temp_train)