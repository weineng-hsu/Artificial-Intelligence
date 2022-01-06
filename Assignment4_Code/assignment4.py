import numpy as np
import math

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
			#print(s)
			i = s % y.size
			if (i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

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
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		output = np.dot(input, self.w) + self.b
		self.a = np.transpose(input)
		return output

	def backward(self, gradients):
		# Write backward pass here
		inputArray = np.tile(self.a, (1, self.w.shape[1]))
		toReturn = np.transpose(inputArray * gradients)
		self.w = self.w - self.lr * inputArray * gradients
		self.b = self.b - self.lr * np.sum(gradients)
		return toReturn

class Sigmoid:
	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		output = 1 / (1 + np.exp(-1 * input))
		self.z = output
		return output

	def backward(self, gradients):
		#Write backward pass here
		return (self.z * (1 - self.z)) * gradients


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum(axis=1))

	def findRandomCentroid(self, X):
		centroidArray = np.random.randint(np.shape(X)[0], size=self.k)
		#checking no duplicate centroids in centroidArray
		unique, counts = np.unique(centroidArray, return_counts=True)
		while np.shape(counts)[0] < self.k:
			centroidArray = np.random.randint(np.shape(X)[0], size=self.k)
			unique, counts = np.unique(centroidArray, return_counts=True)
		return centroidArray

	def assignToCluster(self, X, centerData):
		clusterArray = np.full(X.shape[0], -1)
		for index, example in enumerate(X):
			clusterArray[index] = np.argmin(self.distance(centerData, example)) + 1
		return clusterArray

	def calculateNewCenter(self, X, clusterArray):
		#create a mean array with size = k, fill with 0
		clusterMean = np.full((self.k, X.shape[1]), 0.0)
		#loop through training data, if data belong to a cluster add it into the corresponding array position
		for index, sample in enumerate(X):
			clusterMean[clusterArray[index] - 1] += sample
		#loop through mean array, compute the mean
		for i in range(self.k):
			clusterMean[i] /= np.count_nonzero(clusterArray == i + 1)
		return clusterMean

	def train(self, X):
		#training logic here
		#input is array of features (no labels)
		for loop in range(self.t):
			if (loop == 0):
				centroidArray = self.findRandomCentroid(X)
				self.cluster = self.assignToCluster(X, X[centroidArray])
				continue
			newCenter = self.calculateNewCenter(X, self.cluster)
			self.cluster = self.assignToCluster(X, newCenter)
			loop += 1
		return self.cluster
		#return array with cluster id corresponding to each item in dataset


class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def distanceMatrix(self, X):
		distanceMatrix = np.full((X.shape[0], X.shape[0]), -1.0)
		for x in range(X.shape[0]):
			for y in range(X.shape[0]):
				if (x >= y):
					distanceMatrix[x][y] = np.nan
					continue
				distanceMatrix[x][y] = self.distance(X[x], X[y])
		return distanceMatrix

	def findClosetAndMerge(self, cluster, distanceMatrix):
		min = np.nanmin(distanceMatrix)
		toMerge = np.where(distanceMatrix == min)
		#print(toMerge)
		#distanceMatrix[toMerge[0][0]][toMerge[0][1]] = np.nan
		#distanceMatrix[toMerge[0][1]][toMerge[0][0]] = np.nan
		cluster1 = cluster[toMerge[0][0]]
		cluster2 = cluster[toMerge[1][0]]
		indexOfCluster1 = np.where(cluster == cluster1)
		indexOfCluster2 = np.where(cluster == cluster2)
		smallerCluster = indexOfCluster1
		changeTo = cluster2
		if (np.shape(indexOfCluster1)[1] > np.shape(indexOfCluster2)[1]):
			smallerCluster = indexOfCluster2
			changeTo = cluster1
		cluster[smallerCluster] = changeTo
		for x in indexOfCluster1[0]:
			for y in indexOfCluster2[0]:
				#print(x, y)
				distanceMatrix[x][y] = np.nan
				distanceMatrix[y][x] = np.nan

	def train(self, X):
		#training logic here
		self.cluster = np.arange(np.shape(X)[0])
		self.distanceMatrix = self.distanceMatrix(X)
		while(np.shape(np.unique(self.cluster))[0] > self.k):
			self.findClosetAndMerge(self.cluster, self.distanceMatrix)
		return self.cluster
		#return array with cluster id corresponding to each item in dataset

