import numpy
import numpy as np


### Assignment 3 ###
### Weineng Hsu ###
### wnh215 ###

class KNN:
    def __init__(self, k):
        # KNN state here
        # Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB) ** 2
        return np.sqrt(diffs.sum())

    def findIndexKNearest(self, distance):
        kNearest = np.full(self.k, -1)
        disModify = np.copy(distance)
        maxDis = np.argmax(distance)
        for i in range(self.k):
            iNearest = np.argmin(disModify)
            kNearest[i] = iNearest
            disModify[iNearest] = distance[maxDis]
        return kNearest

    def train(self, X, y):
        # training logic here
        # input is an array of features and labels
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        predict = np.empty(X.shape[0])
        for index_X in range(X.shape[0]):
            xDistance = np.full(self.X_train.shape[0], 0)
            for train in range(self.X_train.shape[0]):
                xDistance[train] = self.distance(X[index_X], self.X_train[train])
            kNearestIndex = self.findIndexKNearest(xDistance)
            neighbors = np.empty(self.k, np.int)
            for i in range(self.k):
                neighbors[i] = self.y_train[kNearestIndex[i]]
            predict[index_X] = np.bincount(neighbors).argmax()
        return predict


class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    def yValueWithActivation(self, row_X):
        sum = self.b + np.sum(np.multiply(row_X, self.w))
        if sum > 0:
            return 1
        else:
            return 0

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels
        index = 0
        for step in range(steps):
            if index > X.shape[0] - 1:
                index = 0
            predict = np.full(y.shape[0], 0)
            predict[index] = self.yValueWithActivation(X[index])
            error = y - predict
            self.w = self.w + self.lr * error[index] * X[index]
            index += 1

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        predict = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            predict[i] = self.yValueWithActivation(X[i])
        return predict


class ID3:
    def __init__(self, nbins, data_range):
        # Decision tree state here
        # Feel free to add methods
        self.bin_size = nbins
        self.range = data_range
        self.tree = None

    def preprocess(self, data):
        # Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size * norm_data).astype(int)
        return categorical_data

    def train(self, X, y):
        # training logic here
        # input is array of features and labels
        categorical_data = self.preprocess(X)
        self.tree = self.buildTree(categorical_data, categorical_data, np.arange(0, X.shape[1]), y)

    def buildTree(self, splitdata, categorical_data, features, y, parent=None):
        if np.unique(y).shape[0] == 1:
            return np.unique(y)[0]
        elif len(splitdata) == 0:
            return parent
        elif features.shape[0] == 0:
            return parent
        currentMaxY = np.argmax(np.unique(y, return_counts=True)[1])
        parent = np.unique(y)[currentMaxY]
        entropyList = np.full(features.shape[0], -1.0)
        for x in range(features.shape[0]):
            featuresIndex = features[x]
            entropyList[x] = self.entropyAttribute(splitdata[:, featuresIndex], y)
        maxGainAttribute = self.findMaxGain(entropyList, self.entropyS(y), features)
        newFeatures = np.full(features.shape[0] - 1, -1)
        featureIndex = 0
        for f in features:
            if f == maxGainAttribute:
                continue
            if newFeatures.shape[0] == 0:
                break
            newFeatures[featureIndex] = f
            featureIndex += 1
        tree = {maxGainAttribute: {}}
        children = np.unique(categorical_data[:, maxGainAttribute])
        for child in children:
            newSplitdata, newY = self.splitData(splitdata, y, maxGainAttribute, child)
            subtree = self.buildTree(newSplitdata, categorical_data, newFeatures, newY, parent)
            tree[maxGainAttribute][child] = subtree
        return tree

    def splitData(self, X, y, node, nodeValue):
        wantedRows = np.argwhere(X[:, node] == nodeValue)
        split = np.full((wantedRows.shape[0], X.shape[1]), 0)
        splitTarget = np.full(wantedRows.shape[0], -1)
        index = 0
        for row in np.argwhere(X[:, node] == nodeValue):
            split[index] = X[row[0]:row[0] + 1, :]
            splitTarget[index] = y[row[0]]
            index += 1
        return split, splitTarget

    def findMaxGain(self, entropy, entropyS, features):
        gains = entropy * -1 + entropyS
        return features[np.argmax(gains)]

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        categorical_data = self.preprocess(X)
        results = np.full(X.shape[0], -1)
        for row in range(X.shape[0]):
            for key in self.tree.keys():
                results[row] = self.traverseTree(self.tree.get(key), key, categorical_data[row])
        return results

    def traverseTree(self, tree, node, categorical_data_row):
        if isinstance(tree.get(categorical_data_row[node]), dict):
            key = list(tree.get(categorical_data_row[node]).keys())[0]
            nextTree = (tree.get(categorical_data_row[node])).get(key)
            return self.traverseTree(nextTree, categorical_data_row[key], categorical_data_row)
        return tree.get(categorical_data_row[node])

    def entropy(self, yCounts):
        if yCounts.shape[0] == 1:
            return 0
        a = yCounts[0]
        b = yCounts[1]
        return a / (a + b) * np.log2(a / (a + b)) + b / (a + b) * np.log2(b / (a + b))

    def entropyAttribute(self, X, y):
        uniqueClasses, countsInClasses = np.unique(X, return_counts=True)
        entropy = 0
        totalClasses = np.sum(countsInClasses)
        for index, type in enumerate(uniqueClasses):
            yWithType = np.array(y)[np.where(X == type)]
            uniqueY, yCountsInType = np.unique(yWithType, return_counts=True)
            if uniqueY.shape[0] == 1:
                continue
            entropy -= countsInClasses[index] / totalClasses * self.entropy(yCountsInType)
        return entropy

    def entropyS(self, y):
        uniqueClasses, countsInClasses = np.unique(y, return_counts=True)
        entropy = 0
        totalCounts = y.shape[0]
        for counts in countsInClasses:
            if counts == 0:
                continue
            entropy -= (counts / totalCounts) * np.log2(counts / totalCounts)
        return entropy


### Assignment 4 ###

class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if (i == 0):
                X, y = self.shuffle(X, y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            # print(loss)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        # Write forward pass here
        return None

    def backward(self, gradients):
        # Write backward pass here
        return None


class Sigmoid:

    def __init__(self):
        None

    def forward(self, input):
        # Write forward pass here
        return None

    def backward(self, gradients):
        # Write backward pass here
        return None


class K_MEANS:

    def __init__(self, k, t):
        # k_means state here
        # Feel free to add methods
        # t is max number of iterations
        # k is the number of clusters
        self.k = k
        self.t = t

    def distance(self, centroids, datapoint):
        diffs = (centroids - datapoint) ** 2
        return np.sqrt(diffs.sum(axis=1))

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        return self.cluster
    # return array with cluster id corresponding to each item in dataset


class AGNES:
    # Use single link method(distance between cluster a and b = distance between closest
    # members of clusters a and b
    def __init__(self, k):
        # agnes state here
        # Feel free to add methods
        # k is the number of clusters
        self.k = k

    def distance(self, a, b):
        diffs = (a - b) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        return self.cluster
    # return array with cluster id corresponding to each item in dataset
