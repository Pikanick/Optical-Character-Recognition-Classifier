# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
#


# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)   # Stops Program and alerts illegal Labels
        self.weights = weights

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector of values).
        """

        self.features = trainingData[0].keys() # could be useful later

        # Feature vector is a flatted version of an image here. eg 28px*28px=784 long vector
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                # util.raiseNotDefined()
                #compute score for each label:
                scores = util.Counter()  # to store scores of each label.
                for label in self.legalLabels:  # use training data to adjust the weights based on classification errors
                    scores[label] = self.weights[label] * trainingData[i]  # score(x,y) = sum(W*F)

                #find the most optimum label:

                prediction = scores.argMax()  # highest value is the guess
                TrueLabel = trainingLabels[i]

                # update weight if necessary:
                if prediction != TrueLabel:  # update the weights so that we can make better guesses in the next iteration.
                    # w^y=w^y+f
                    self.weights[TrueLabel] += trainingData[i]  #Increase the weights for the actual label by the feature vector.
                    # w^y=w^y-f
                    self.weights[prediction] -= trainingData[i]  #Decrease the weights for the guessed label by the feature vector.


        print("finished training")


    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
      
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
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

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        weights = self.weights[label]
        featuresWeights = weights.sortedKeys()[:100]

        return featuresWeights
