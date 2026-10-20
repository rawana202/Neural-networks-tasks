import numpy as np

class Adaline:
    def __init__(self, learningRate=0.001, iterationsNumber=100):
        self.learningRate = learningRate
        self.iterationsNumber = iterationsNumber

    def fit(self, features, targets, mse_threshold=0.01):
        nOfSamples, nOfFeatures = features.shape
        self.bias = 0.0
        self.threshold = mse_threshold 
        self.weights = np.random.randn(nOfFeatures) * 0.01

        for j in range(self.iterationsNumber):
            output = np.dot(features, self.weights) + self.bias
            errors = targets - output

            mse = np.mean(errors ** 2)
            if self.threshold is not None and mse <= self.threshold:
                break

            self.weights += self.learningRate * np.dot(features.T, errors)
            self.bias += self.learningRate * np.sum(errors)  
        return

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        predictions = []
        for x in output:
            if x >= 0.0:
                predictions.append(1)
            else:
                predictions.append(-1)
        return predictions