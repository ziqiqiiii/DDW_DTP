import numpy as np
from utils.GradientDescentUtils import compute_cost, predict_y, update_parameters

def initialize(dim):
  w1 = np.random.rand(dim)
  w0 = np.random.rand()
  return w1, w0

def run_gradient_descent(X,Y,alpha,max_iterations,stopping_threshold = 1e-6):
  # dims = 1
  # if len(X.shape)>1:
  dims = X.shape[1]
  w1,w0=initialize(dims)
  previous_cost = None
  cost_history = np.zeros(max_iterations)
  for itr in range(max_iterations):
    y_hat=predict_y(X,w1,w0)
    cost=compute_cost(X,Y,y_hat)
    # early stopping criteria
    if previous_cost and abs(previous_cost-cost)<=stopping_threshold:
      break
    cost_history[itr]=cost
    previous_cost = cost
    old_w1=w1
    old_w0=w0
    w0,w1=update_parameters(X,Y,y_hat,cost,old_w0,old_w1,alpha)

  return w0,w1,cost_history

# Reference for Gradient Descent 
# https://medium.com/@pritioli/implementing-linear-regression-from-scratch-747343634494
