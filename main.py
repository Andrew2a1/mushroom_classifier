import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

from data_converter import DataConverter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt


def split_data(data, train_percentage):
    return train_test_split(data, train_size=int(len(data)*train_percentage))

data = pd.read_csv("primary_data.csv", sep=';')

#konfigurowalny podzia zbioru


# Jest 20 cech, cechy są w formie tekstowej i wymagają przetworzenia na liczby
# zliczanie próbek w klasie

class_count = {}
for probe in data.values:
    class_name = probe[0]
    if class_name in class_count:
        class_count[class_name] += 1
    else:
        class_count[class_name] = 1

print("\n", class_count)
plt.bar(class_count.keys(), class_count.values())


# zbiór danych nie jest znormalizowany poniewaz posiada on litery co stanowi przeszkodę w drodzę do kwalifikacji, więc wymaga ono takiej czynności.
# strin na float oraz normalizacja danych

converter = DataConverter()
converted = converter.convert_data(data)
normalized = preprocessing.normalize(converted)

labels = data["family"].to_list()
classes = list(set(labels))

label_numbers = np.reshape(np.array([classes.index(label) for label in labels]), (173, 1))

# podział zbioru na testowego na testowy i walidacyjne

normalized_with_labels = np.append(normalized, label_numbers, 1)
train, test = split_data(normalized_with_labels, 0.8)

# Wybór najlepszych cech "L1-based feature selection"

lsvc = LinearSVC().fit(train[:, :-1], train[:, -1].astype('int'))
model = SelectFromModel(lsvc, prefit=True)

train_new = model.transform(train[:, :-1])
test_new = model.transform(test[:, :-1])

# Wybrano 11 najlepszych cech
print(train_new.shape, train.shape)

# Klasyfikacja
# NN,

nbrs = NearestNeighbors(n_neighbors=1).fit(train_new)

valid = 0
for test_idx in range(len(test_new)):
    _, indices = nbrs.kneighbors(test_new[test_idx].reshape(1, -1))

    print(indices)
    print(test[indices[0]])

    estimated_family = test[indices[0]][-1]
    family = test[test_idx][-1]

    if estimated_family == family:
        valid += 1

print(f"Accuracy: {valid/len(test_new)*100}%")




# k-NN,
# NM i
# k-NM.

