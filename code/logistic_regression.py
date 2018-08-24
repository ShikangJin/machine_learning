from sklearn.linear_model.logistic import LogisticRegression
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    weight = classifier.coef_
    result = classifier.predict(X_test)

    for index in range(0, len(weight)):
        plt.subplot(1, 10, index + 1)
        plt.axis('off')
        plt.imshow(weight[index].reshape(8, 8))

    for index in range(0, len(result)):
        print("label group: " + repr(result[index]) + ", real digit: " + repr(y_test[index]))
    plt.show()
