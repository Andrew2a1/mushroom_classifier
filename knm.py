# Sredni wektor
from collections import Counter

import numpy as np
from scipy.spatial import distance


def knm(X, y, test_x, k=5):
    classes = set(y)

    distances = {}
    for class_number in classes:
        probes = X[y == class_number]
        avg = np.mean(probes, 0)

        C = np.cov(probes, rowvar=False)
        try:
            VI = np.linalg.inv(C)
            dist = distance.mahalanobis(test_x, avg, VI)
        except np.linalg.LinAlgError:
            dist = distance.euclidean(test_x, avg)

        distances[class_number] = dist

    sorted_dists = sorted(distances.items(), key=lambda d: d[1])
    nearest_classes = [c[0] for c in sorted_dists[:k]]

    return Counter(nearest_classes).most_common(1)[0][0]

def knm_fit(X, y, test_X, k=5):
    results = []
    for idx in range(len(test_X)):
        results.append(knm(X, y, test_X[idx], k))
    return results
