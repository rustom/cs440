'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''

    num_train = len(train_images)
    distances = np.zeros(num_train)

    for i in range(num_train):
        distances[i] = np.sum((image - train_images[i]) ** 2)

    indices = np.argsort(distances)
    neighbors = train_images[indices[:k]]
    labels = train_labels[indices[:k]]

    return neighbors, labels



def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    
    num_dev = len(dev_images)
    hypotheses = np.zeros(num_dev)
    scores = np.zeros(num_dev)

    for i in range(num_dev):
        neighbors, labels = k_nearest_neighbors(dev_images[i], train_images, train_labels, k)
        
        # if tie occurs, predict negative class

        num_positive = np.sum(labels)
        if num_positive > k / 2:
            hypotheses[i] = 1
            scores[i] = num_positive
        else:
            hypotheses[i] = 0
            scores[i] = k - num_positive

    return hypotheses, scores

def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    confusions = np.zeros((2, 2))
    for i in range(len(hypotheses)):
        confusions[references[i] * 1][hypotheses[i] * 1] += 1

    accuracy = (confusions[0][0] + confusions[1][1]) / np.sum(confusions)
    f1 = 2 * confusions[1][1] / (2 * confusions[1][1] + confusions[0][1] + confusions[1][0])

    return confusions, accuracy, f1
