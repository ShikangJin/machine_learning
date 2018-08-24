from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

if __name__ == "__main__":

    digits = datasets.load_digits()

    for index in range(0, 10):
        plt.subplot(3, 10, index + 1)
        plt.axis('off')
        plt.imshow(digits.data[index].reshape(8, 8))

    kmeans = KMeans(n_clusters=10)
    result = kmeans.fit_predict(digits.data)
    centroid = kmeans.cluster_centers_
    label = kmeans.labels_

    if len(digits.target) != len(result):  # must be the same
        print("ERROR: The amount of label is not equal to the amount of real data")
        sys.exit()

    for index in range(0, len(result)):
        print("label group: " + repr(result[index]) + ", real digit: " + repr(digits.target[index]))

    for index in range(0, len(centroid)):
        plt.subplot(3, 10, index + 11)
        plt.axis('off')
        plt.imshow(centroid[index].reshape(8, 8))

    plt.show()





