import numpy as np

def compute_cost(X,Y, y_hat):
    m = len(Y)
    cost = (1/(2*m)) * np.sum(np.square(y_hat - Y))
    return cost

def predict_y(X,w1,w0):
  if len(w1)==1:
    w1 = w1[0]
    return X*w1+w0
  return np.dot(X,w1)+w0

def update_parameters(X,Y,y_hat,cost,w0,w1,learning_rate):
  m = len(Y)
  db=(np.sum(y_hat-Y))/m
  dw=np.dot(y_hat-Y,X)/m

  w0_new=w0-learning_rate*db
  w1_new=w1-learning_rate*dw
  return w0_new,w1_new
