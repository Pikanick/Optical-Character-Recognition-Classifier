# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        bestAccuracyCount = -1  # best accuracy so far on validation set
        cGrid.sort(reverse=True)
        bestParams = cGrid[0]

        bestWeights = None

        for C in cGrid:
            # Prepare an empty weight vector for each possible class label using a dictionary where legal labels is the key.
            self.weights = {label: util.Counter() for label in self.legalLabels}
            for iteration in range(self.max_iterations):
                for i in range(len(trainingData)):
                    scores = util.Counter()
                    for label in self.legalLabels:
                        scores[label] = self.weights[label] * trainingData[i]  # compute the scores for each class using the current weights and feature vectors like perceptron
                    prediction = scores.argMax()
                    TrueLabel = trainingLabels[i]
                    if prediction != TrueLabel:
                        tau = min(C, ((self.weights[prediction] - self.weights[TrueLabel]) * trainingData[i] + 1.0) / (
                                    2.0 * (trainingData[i] * trainingData[i]))) # update tau based on formula from notes
                        tauFeature = trainingData[i].copy()
                        tauFeature.divideAll(1.0 / tau)  # Scaled copy of feature vector.
                        self.weights[TrueLabel] += tauFeature
                        self.weights[prediction] -= tauFeature

            correct = 0
            for i in range(len(validationData)):
                scores = util.Counter()
                for label in self.legalLabels:
                    scores[label] = self.weights[label] * validationData[i]
                prediction = scores.argMax()
                if prediction == validationLabels[i]:
                    correct += 1
            accuracy = correct / len(validationLabels)
            if accuracy > bestAccuracyCount:
                bestAccuracyCount = accuracy
                bestWeights = self.weights.copy()  # ensure copy is made rather than simply making a reference to self.weights.
                bestParams = C

        self.weights = bestWeights
        print("finished training. Best cGrid param = ", bestParams)

        return bestAccuracyCount

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []

        for datum in data:
            scores = util.Counter()
            for label in self.legalLabels:  # computing the scores for each label to predict the label of a new data instance.
                scores[label] = self.weights[label] * datum
            guesses.append(scores.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        weights = self.weights[label]
        featuresWeights = weights.sortedKeys()[:100]
        return featuresWeights
