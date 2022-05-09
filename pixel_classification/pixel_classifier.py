'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np

class PixelClassifier():
  def __init__(self,lr = 0.01, iteration = 10000,classes = 3, features = 3, pretrained = False):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.learningRate = lr  # learning rate
    self.iterNumber = iteration # number of iteration

    if pretrained: # assign weight value from training weight file
      self.w = np.load('pixel_weight.npy') # load training weight file (approach #1)
    else: # Mannually assign value (copied from training) for gradescope check (approach #2)
      self.w = np.array([[ 2.75162998, -1.4468946 , -1.51191691], [-1.21596619,  2.48733989, -1.65745635], [-1.13724369, -1.41401972,  2.15218174]])

    pass
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE

    w = self.w 
    #flatten =  X.reshape(X.shape[0],-1)
    #calculate the softmax
    scores = np.dot(X, w.T)
    probs = self.softmax(scores)
    y = np.argmax(probs, axis=1).reshape((-1,1))+1
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

  def softmax(self, X):
    # calculate the softmax
    exp = np.exp(X)
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    softmax = exp / sum_exp
    return softmax

  def accuracy(self,pred, true):
    return sum(true == pred) / len(pred)

  def predict(self, test_data, w):
    scores = np.dot(test_data, w.T)
    probs = self.softmax(scores)
    return np.argmax(probs, axis=1).reshape((-1, 1))


  def train(self, X, Y):

    # Change Y to one-hot
    true = Y.copy()
    temp = np.zeros((Y.size, Y.max()))
    temp[np.arange(Y.shape[0]), Y-1] = 1
    Y = temp
    m = X.shape[1]  # features of x
    n = Y.shape[1]  # # of y class
    M = X.shape[0]  # # of samples
    w = np.random.randn(n, m) # weight
    all_loss = list()
    # np.dot(X,w)
    for t in range(self.iterNumber):
      y1 = self.softmax(np.dot(X, w.T))
      loss = - (1.0 / M) * np.sum(Y * np.log(y1))
      # loss is for whole dataset loss
      all_loss.append(loss)
      # los in dw is for updating the parameter. It is the loss for each parameter
      dw = -(1.0 / M) * np.dot((Y - y1).T, X) + self.learningRate * w
      w -= self.learningRate * dw
    prediction = self.predict(X, w)
    print('training accuracy:',self.accuracy(prediction.ravel(),true-1))
    np.save('weight.npy', w)
    self.w = w
    pass