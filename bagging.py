import numpy as np
import math
import random

def draw_bagging_indices(n_features, n_estimators):
    bagging_features_indices = list()
    for i in range(n_estimators):
        bagging_features_indices.append(random.sample(range(n_features), math.ceil(math.sqrt(n_features))))
    return bagging_features_indices