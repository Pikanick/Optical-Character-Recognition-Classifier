# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import mira
import samples
import sys
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
    - Diagonal Gradients: Captures the gradient differences along the diagonals of the image, rather than just the horizontal and vertical directions.
    - Central Pixel Intensity: Finds the sum of pixel intensities in the central region of the image.
    - Perimeter Pixel Count: Counts the number of active pixels along the perimeter of the image.
    - Sum of Pixel Intensities for Rows and Columns: Calculates the sum of pixel intensities for each row and column.
    - Zoning: Divides the image into a 3x3 grid and calculate the sum of pixel intensities in each zone.
    - Aspect Ratio: Computes the aspect ratio of the bounding box around the digit.
    - Symmetry: Measures the vertical and horizontal symmetry of the digit.
    ##
    """
    features = basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE to extract and add enhanced features to features list ***"
    # util.raiseNotDefined()

    # Feature: Diagonal gradients
    for x in range(1, DIGIT_DATUM_WIDTH):
        for y in range(1, DIGIT_DATUM_HEIGHT - 1):  # Ensure y + 1 is within bounds
            if x - 1 >= 0 and y - 1 >= 0:  # Ensure x - 1 and y - 1 are within bounds
                features[("diag1", x, y)] = int(datum.getPixel(x, y) > datum.getPixel(x - 1, y - 1))
            if x - 1 >= 0 and y + 1 < DIGIT_DATUM_HEIGHT:  # Ensure x - 1 and y + 1 are within bounds
                features[("diag2", x, y)] = int(datum.getPixel(x, y) > datum.getPixel(x - 1, y + 1))

    # Feature: Central pixel intensity
    central_region_size = 6  # Define the size of the central region
    central_x_start = (DIGIT_DATUM_WIDTH - central_region_size) // 2
    central_y_start = (DIGIT_DATUM_HEIGHT - central_region_size) // 2
    central_intensity = 0
    for x in range(central_x_start, central_x_start + central_region_size):
        for y in range(central_y_start, central_y_start + central_region_size):
            central_intensity += datum.getPixel(x, y)
    features["central_intensity"] = central_intensity

    # Feature: Perimeter pixel count
    perimeter_count = 0
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if x == 0 or x == DIGIT_DATUM_WIDTH - 1 or y == 0 or y == DIGIT_DATUM_HEIGHT - 1:
                if datum.getPixel(x, y) > 0:
                    perimeter_count += 1
    features["perimeter_count"] = perimeter_count

    # Feature: Sum of pixel intensities for rows and columns
    row_sums = [0] * DIGIT_DATUM_HEIGHT
    col_sums = [0] * DIGIT_DATUM_WIDTH
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            pixel = datum.getPixel(x, y)
            row_sums[y] += pixel
            col_sums[x] += pixel

    for y in range(DIGIT_DATUM_HEIGHT):
        features[f"row_sum_{y}"] = row_sums[y]
    for x in range(DIGIT_DATUM_WIDTH):
        features[f"col_sum_{x}"] = col_sums[x]

    # Feature: Zoning (3x3 grid)
    zone_width = DIGIT_DATUM_WIDTH // 3
    zone_height = DIGIT_DATUM_HEIGHT // 3
    for i in range(3):
        for j in range(3):
            zone_sum = 0
            for x in range(i * zone_width, (i + 1) * zone_width):
                for y in range(j * zone_height, (j + 1) * zone_height):
                    zone_sum += datum.getPixel(x, y)
            features[f"zone_{i}_{j}"] = zone_sum

    # Feature: Aspect Ratio
    def bounding_box():
        min_x, max_x, min_y, max_y = DIGIT_DATUM_WIDTH, 0, DIGIT_DATUM_HEIGHT, 0
        for x in range(DIGIT_DATUM_WIDTH):
            for y in range(DIGIT_DATUM_HEIGHT):
                if datum.getPixel(x, y) > 0:
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
        return (max_x - min_x + 1), (max_y - min_y + 1)

    bbox_width, bbox_height = bounding_box()
    features['aspect_ratio'] = bbox_width / bbox_height if bbox_height > 0 else 0

    # Feature: Symmetry
    def symmetry():
        vertical_symmetry = 0
        horizontal_symmetry = 0
        for x in range(DIGIT_DATUM_WIDTH // 2):
            for y in range(DIGIT_DATUM_HEIGHT):
                if datum.getPixel(x, y) == datum.getPixel(DIGIT_DATUM_WIDTH - 1 - x, y):
                    vertical_symmetry += 1
        for y in range(DIGIT_DATUM_HEIGHT // 2):
            for x in range(DIGIT_DATUM_WIDTH):
                if datum.getPixel(x, y) == datum.getPixel(x, DIGIT_DATUM_HEIGHT - 1 - y):
                    horizontal_symmetry += 1
        return vertical_symmetry, horizontal_symmetry

    vert_sym, horiz_sym = symmetry()
    features['vertical_symmetry'] = vert_sym / (DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT // 2)
    features['horizontal_symmetry'] = horiz_sym / (DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT // 2)

    return features


def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print("===================================")
            print("Mistake on example %d" % i)
            print("Predicted %d; truth is %d" % (prediction, truth))
            print("Image: ")
            print(rawTestData[i])
            break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print (image)

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples, using smoothing 
                  factor k=2.5 and uses enhancedFeatureExtractorDigit() to extract new features 
                  for digit datum, and add them to the feature set, it also uses 
                 """

def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print ("Doing classification")
    print ("--------------------")
    print ("data:\t\t" + options.data)
    print ("classifier:\t\t" + options.classifier)
    print ("using enhanced features?:\t" + str(options.features))
    print ("training set size:\t" + str(options.training))
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
    else:
        print("Unknown dataset"), options.data
        print(USAGE_STRING)
        sys.exit(2)

    legalLabels = range(10)

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print ("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print (USAGE_STRING)
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print ("using automatic tuning for naivebayes")
            classifier.automaticTuning = True
        else:
            print ("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
    elif(options.classifier == "perceptron"):
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print ("using automatic tuning for MIRA")
            classifier.automaticTuning = True
        else:
            print ("using default C=0.001 for MIRA")
    else:
        print ("Unknown classifier:", options.classifier)
        print (USAGE_STRING)


        sys.exit(2)

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options


# Main harness code

def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print("Extracting features...")
    trainingData = list(map(featureFunction, rawTrainingData))
    validationData = list(map(featureFunction, rawValidationData))
    testData = list(map(featureFunction, rawTestData))

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print(string3)
        printImage(features_odds)

    if(options.weights and options.classifier == "perceptron"):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

        from answers import q2
        print("Question 2's answer: ", q2())

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
