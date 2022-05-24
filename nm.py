# Sredni wektor
import numpy as np

def avg_vect(vectors):
    return np.mean(vectors, axis=0)

# Macierz kowariancji
def cov(vectors):
    n, _ = vectors.shape
    diff = vectors - avg_vect(vectors)
    return np.transpose(diff) @ diff/(n - 1)

def machdist(p1, data):
    C = cov(data)
    p2 = avg_vect(data)
    return np.sqrt(np.transpose(p1 - p2) @ np.linalg.inv(C) @ (p1 - p2))

# Probki dla danej klasy
def for_class(data, class_number):
    return data[data[:, 0] == class_number]

# Dane cechy
def for_features(data, start, end):
    return data[:, start:end]

def nm_machdist(train, test_probe, feature_start=0, feature_end=-1):
    classes = set(int(x[0]) for x in train)
    distances = {}
    
    for class_number in classes:
        distances[class_number] = machdist(test_probe[feature_start:feature_end], for_features(for_class(train, class_number), feature_start, feature_end))
        
    return min(distances.items(), key=lambda d: d[1])

def NM_MACH(test, train):
    classified = [nm_machdist(train, test)[0] for test in test]
    accuracy = len([1 for n, item in enumerate(test) if classified[n] == item[0]])/len(test)
    print(f"accuracy = {accuracy*100}%")

    return classified