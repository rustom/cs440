'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''

    max_word_0 = 0
    max_word_1 = 0
    for text in texts:
        max_word_0 = max(max_word_0, text.count(word0))
        max_word_1 = max(max_word_1, text.count(word1))

    Pjoint = np.zeros((max_word_0+1, max_word_1+1))

    for text in texts:
        Pjoint[text.count(word0), text.count(word1)] += 1

    Pjoint /= Pjoint.sum()

    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''

    if index == 0:
        Pmarginal = np.zeros(Pjoint.shape[0])
    else:
        Pmarginal = np.zeros(Pjoint.shape[1])

    if index == 0:
        Pmarginal = Pjoint.sum(axis=1)
    else:
        Pmarginal = Pjoint.sum(axis=0)

    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''

    Pcond = np.zeros(Pjoint.shape)
    Pcond = Pjoint / Pmarginal[:,np.newaxis]

    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''

    indices = np.arange(P.shape[0])
    mu = (indices * P).sum()

    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''

    indices = np.arange(P.shape[0])

    mu = mean_from_distribution(P)
    var = ((indices-mu)**2 * P).sum()

    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''

    indices0 = np.arange(P.shape[0])
    indices1 = np.arange(P.shape[1])

    mu0 = mean_from_distribution(P.sum(axis=1))
    mu1 = mean_from_distribution(P.sum(axis=0))

    covar = ((indices0-mu0)[:,np.newaxis] * (indices1-mu1)[np.newaxis,:] * P).sum()
    
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''

    indices0 = np.arange(P.shape[0])
    indices1 = np.arange(P.shape[1])

    expected = (f(indices0[:,np.newaxis], indices1[np.newaxis,:]) * P).sum()

    return expected
    
