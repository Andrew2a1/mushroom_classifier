import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from data_converter import DataConverter
from knm import knm_fit

# Wczytanie danych

data = pd.read_csv("primary_data.csv", sep=';')

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

def split_data(data, train_percentage):
    return train_test_split(data, train_size=int(len(data)*train_percentage))

normalized_with_labels = np.append(normalized, label_numbers, 1)
train, test = split_data(normalized_with_labels, 0.8)

# Wybór najlepszych cech "L1-based feature selection"

lsvc = LinearSVC(dual=False).fit(train[:, :-1], train[:, -1].astype('int'))
model = SelectFromModel(lsvc, prefit=True)

train_new = model.transform(train[:, :-1])
test_new = model.transform(test[:, :-1])

# Wybrano 11 najlepszych cech
print(train_new.shape, train.shape)

# Klasyfikacja
# NN,

labels_int = train[:, -1].astype('int')

clf = KNeighborsClassifier(1)
clf.fit(train_new, labels_int)
predictions = clf.predict(test_new)

print("NN")
print(f"Accuracy: {round(100*sum(test[:, -1] == predictions)/len(predictions), 3)}%")

# k-NN,

clf = KNeighborsClassifier(10)
clf.fit(train_new, labels_int)
predictions = clf.predict(test_new)

print("KNN")
print(f"Accuracy: {round(100*sum(test[:, -1] == predictions)/len(predictions), 3)}%")

# NM i

print("NM")
predictions = knm_fit(train_new, labels_int, test_new, k=1)
print(f"Accuracy: {round(100*sum(test[:, -1] == predictions)/len(predictions), 3)}%")

# k-NM.

print("k-NM")
predictions = knm_fit(train_new, labels_int, test_new, k=12)
print(f"Accuracy: {round(100*sum(test[:, -1] == predictions)/len(predictions), 3)}%")
