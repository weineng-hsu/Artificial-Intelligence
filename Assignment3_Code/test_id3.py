import assignment3 as models
import numpy as np
import sys
import numpy as np

#test id3 with small dataset

#use the example from class - to go to the park
# feat 1 -> [1 = alone, 2 = friend]
# feat 2 -> [1 = sunny, 2 = cloudy, 3 = rainy]
# feat 3 -> [1 = long, 2 = short]
#X_train = np.array([[1,1,1],[2,1,2],[1,2,2],[2,3,1],[2,3,2],[2,1,2],[1,2,1],[2,1,1],[1,1,1]])
#y_train = np.array([0,1,1,0,0,1,1,0,0])
#X_test = np.array([[1,1,2],[2,2,1]])
#y_test = np.array([1,1])
#nbins = 2


# feat 1 -> [1 = rainy, 2 = overcast, 3 = sunny]
# feat 2 -> [1 = hot, 2 = mild, 3 = cool]
# feat 3 -> [1 = high, 2 = normal]
# feat 4 -> [1 = false, 2 = true]
# feat 5 -> [1 = no, 2 = yes]
#X_train = np.array([[1,1,1,1],[1,1,1,2],[2,1,1,1],[3,2,1,1],[3,3,2,1],[3,3,2,2],[2,3,2,2],[1,2,1,1],[1,3,2,1],[3,2,2,1],[1,2,2,2]])
#y_train = np.array([0,0,1,1,1,0,1,0,1,1,1])
#X_test = np.array([[2,2,1,2],[2,1,2,1],[3,2,1,2]])
#y_test = np.array([1,1,0])
#nbins = 3

# feat 1 -> [1 = rainy, 2 = overcast, 3 = sunny]
# feat 2 -> [1 = hot, 2 = mild, 3 = cool]
# feat 3 -> [1 = high, 2 = normal]
# feat 4 -> [1 = weak, 2 = strong]
# feat 5 -> [1 = no, 2 = yes]
#X_train = np.array([[3,1,1,1], [3,1,1,2], [2,1,1,1], [1,2,1,1], [1,3,2,1], [1,3,2,2], [2,3,2,2], [3,2,1,1], [3,3,2,1], [1,2,2,1],[3,2,2,2]])
#y_train = np.array([0,0,1,1,1,0,1,0,1,1,1])
X_train = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0.],[1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0.]],dtype=np.int64)
y_train = np.array([0, 1, 0, 1])

#y_train = np.array([0, 1, 0, 1])
X_test = np.array([[2,2,1,2],[2,1,2,1],[1,2,1,2]])
y_test = np.array([1,1,0])
nbins = 3

#ID3

data_range = (X_train.min(0), X_train.max(0))
id3 = models.ID3(nbins, data_range)

#Train
id3.train(X_train, y_train)
#print(f"TREE: {id3.tree}")

#Evaluate
def evaluate(solutions, real):
	if(solutions.shape != real.shape):
		raise ValueError("Output is wrong shape.")
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

#solutions = id3.predict(X_test)
#print(f"Output: {solutions}")
#print(f"Accuracy: {evaluate(solutions, y_test)*100}%")

