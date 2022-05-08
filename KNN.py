import numpy as np
from collections import Counter

def euclideanDistance(x, y):
    return np.sqrt(np.sum((x-y)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        predictions = [self.singlePredict(x) for x in X]
        return predictions

    def singlePredict(self, x):
        distances = [euclideanDistance(x, x_train) for x_train in self.X_train]
        idxDist = np.argsort(distances)[:self.k]
        nearLabels = [self.y_train[idx] for idx in idxDist]
        most_common = Counter(nearLabels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))
    

        