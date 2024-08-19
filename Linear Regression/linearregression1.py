# Assignment: Linear Regression
# In this assignment you will implement Linear Regression for a very simple
# test case. Please fill into the marked places of the code
#   (1) the cost function
#   (2) the update function for Gradient Descent
#
#   Pieces of code that need to be updated are marked with "HERE YOU ..."
#
# This assignment is kept very simple on purpose to help you familiarize
# with Linear Regression and its implementation using python. Feel free to make some useful tests such as, but not limited to:
# E.g. What happens if the learning rate is too high or too low?
#      Can Linear Regression really find the absolute global minimum?
#      What effect does it have if you change the initial guess for the
#      gradient descent to something completely off?
#      What happens if you are not updating thet0 and thet1
#      "simultaneously" but you are updating both parameters in separate
#      for loops (see below)?
#      You can try to turn this code for Linear Regression into an
#      implementation of Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# You may find helpful the use of cost (one of the costFunction output parameters) to debug this method
# Hint: print("Iteration %d | Cost: %f" % (i, cost))


def gradientDescent(x, y, theta, alpha, m, maxsteps):
    thetaHist = np.empty([maxsteps, 2])
    xTrans = x.transpose()
    for i in range(0, maxsteps):

        yStar = np.dot(x, theta)
        error = yStar - y

        gradient = np.dot(xTrans, error) / m
        theta = theta - alpha * gradient

        thetaHist[i] = theta
        cost, _ = costFunction(x, y, theta)

        # to depbug
        print("Iteration %d | Cost: %f" % (i, cost))

    return theta, thetaHist

# The cost function template is returning two parameters, loss and cost.
# We proposed these two paremeters to facilitate the implementation having not only the cost but also the difference between y and the prediction directly (loss).


def costFunction(x, y, theta):
    m = len(y)
    yStar = np.dot(x, theta)

    cost = np.sum((yStar - y) ** 2) / (2 * m)
    loss = np.sum(np.abs(yStar - y)) / m

    return cost, loss


# Define some training data
# To test your algorithm it is a good idea to start with very simple test data
# where you know the right answer. So let's put all data points on a line first
# Variables x and y represent a (very simple) training set (a dataset with 9 instances)
# Feel free to play with this test data or use a more realistic one.
# NOTE: The column with 1’s included in the variable x is used to facilitate the calculations in the Gradient Descent function
# (do you remember the x_0 to use the matrix form? If not, revise the lecture).
x = np.array([[1, 0], [1, 0.5], [1, 1], [1, 1.5], [
    1, 2], [1, 2.5], [1, 3], [1, 4], [1, 5]])
y = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5])

# Calculate length of training set
m, n = np.shape(x)

# Plot training set
fig = plt.figure(1)  # An empty figure with no axes
plt.plot(x[:, 1], y, 'x')

# Also it is useful for simple test cases to not just run an optimization
# but first to do a systematic search. So let us first calculate the values
# of the cost function for different parameters theta
theta0 = np.arange(-2, 2.01, 0.25)
theta1 = np.arange(-2, 3.01, 0.25)

J = np.zeros((len(theta0), len(theta1)))

for i in range(0, len(theta0)):
    for j in range(0, len(theta1)):

        theta = np.array([theta0[i], theta1[j]])
        cost, _ = costFunction(x, y, theta)
        J[i, j] = cost

theta0, theta1 = np.meshgrid(theta0, theta1)

fig2 = plt.figure(2)
ax = fig2.add_subplot(121, projection="3d")
surf = ax.plot_surface(theta0, theta1, np.transpose(J))
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
ax.set_zlabel('Cost J')
ax.set_title('Cost function Surface plot')

ax = fig2.add_subplot(222)
contour = ax.contour(theta0, theta1, np.transpose(J))
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
ax.set_title('Cost function Contour plot')

fig2.subplots_adjust(bottom=0.1, right=1, top=0.9)

# Here we implement Gradient Descent
alpha = 0.5      # learning parameter
maxsteps = 1000      # number of iterations that the algorithm is running

# First estimates for our parameters
thet = [2, 0]

thet, thetaHist = gradientDescent(x, y, thet, alpha, m, maxsteps)

# Print found optimal values
print("Optimized Theta0 is ", thet[0])
print("Optimized Theta1 is ", thet[1])

# Now let's plot the found solutions of the Gradient Descent algorithms on
# the contour plot of our cost function to see how it approaches the
# desired minimum.
fig3 = plt.figure(3)
plt.contour(theta0, theta1, np.transpose(J))
plt.plot(thetaHist[:, 0], thetaHist[:, 1], 'x')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')

# Finally, let's plot the hypothesis function into our data
xs = np.array([x[0, 1], x[x.shape[0]-1, 1]])
h = np.array([[thet[1] * xs[0] + thet[0]], [thet[1] * xs[1] + thet[0]]])
plt.figure(1)
plt.plot(x[:, 1], y, 'x')  # Data
plt.plot(xs, h, '-o')     # hypothesis function
plt.show()
