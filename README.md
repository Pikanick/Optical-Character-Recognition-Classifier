# Optical Character Recognition Classifier

This repository contains the implementation of three classifiers used for Optical Character Recognition (OCR) of handwritten digit images: **Naïve Bayes Classifier**, **Perceptron Classifier**, and **MIRA (Margin-Infused Relaxed Algorithm) Classifier**. The classifiers are tested on a dataset of scanned handwritten digit images (0-9) and are designed to achieve a high level of accuracy on this task using simple features.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Classifiers](#classifiers)
  - [Naïve Bayes](#naïve-bayes)
  - [Perceptron](#perceptron)
  - [MIRA](#mira)
- [Feature Engineering](#feature-engineering)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Contributors](#contributors)

## Introduction

Optical Character Recognition (OCR) is the process of converting images of typed, handwritten, or printed text into machine-encoded text. This project focuses on designing and implementing three classifiers to accurately recognize handwritten digits from a dataset of digit images. The dataset used in this project is similar to the one used in many commercial OCR systems.

## Project Structure

The project includes the following files:

- **`naiveBayes.py`**: Implementation of the Naïve Bayes classifier.
- **`perceptron.py`**: Implementation of the Perceptron classifier.
- **`mira.py`**: Implementation of the MIRA classifier.
- **`dataClassifier.py`**: Wrapper code that calls the classifiers and analyzes their performance. Also includes an enhanced feature extractor.
- **`answers.py`**: Contains answers to the analysis questions provided in the assignment.
- **`projectParams.py`**: Contains parameters and configurations used throughout the project.
- **`samples.py`**: I/O code to read in the classification data.
- **`util.py`**: Contains utility functions used by the classifiers.
- **`classificationMethod.py`**: Abstract superclass for the classifiers.
  
## Classifiers

### Naïve Bayes

The Naïve Bayes classifier is a probabilistic classifier based on applying Bayes' theorem with strong (naïve) independence assumptions between the features. This classifier is known for its simplicity and effectiveness, especially with large datasets.

### Perceptron

The Perceptron classifier is a type of linear classifier that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. This classifier is an online learning algorithm and updates the weights as it processes each training instance.

### MIRA

MIRA, or Margin-Infused Relaxed Algorithm, is an online learning algorithm that adjusts its weights by a variable step size, aiming to correct its mistakes with the smallest necessary adjustment. This classifier is closely related to support vector machines and is particularly suited for problems with a large margin between classes.

## Feature Engineering

An essential part of this project is the enhancement of features used by the classifiers. While the baseline classifiers use simple per-pixel features (indicating whether a pixel is on or off), the enhanced feature extractor implemented in `dataClassifier.py` adds more sophisticated features to improve classification accuracy. For example, features capturing the number of connected regions of white pixels, loops in the digits, and other characteristics are considered.

## Installation and Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/ocr-classifiers.git
   cd ocr-classifiers
   ```

2. **Run the classifiers:**
   Each classifier can be run with specific configurations provided in the assignment. For example, to run the Perceptron classifier:
   ```bash
   python dataClassifier.py -c perceptron -d digits -t 1000 -w -1 3 -2 6
   ```

3. **Analyze the results:**
   The results of the classifiers, including accuracy and feature weights, can be viewed directly in the terminal or saved to a file.

## Results

The classifiers were tested on a validation and test set of digit images. The Perceptron classifier achieved approximately 82% validation accuracy and 75% test accuracy. The Naïve Bayes and MIRA classifiers showed comparable performance, with potential improvements using enhanced feature extraction.

## Contributors

- **Nicolas Ramirez** - Implementation and analysis.
