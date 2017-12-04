import numpy as np
get_ipython().magic('matplotlib inline')
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def linear_regression():
    points = np.genfromtxt('data.csv', delimiter = ',')
    learning_rate = 0.0001
    init_a = 0
    init_b = 0
    num_iterations = 10000
    print(
        'Start learning at a = {0}, b = {1}, error = {2}'.format(
            init_a,
            init_b,
            compute_error(init_a, init_b, points)
        )
    )
    a, b = gradient_descent(init_a, init_b, points, learning_rate, num_iterations)
    print(
        'End learning at a = {0}, b = {1}, error = {2}'.format(
            a,
            b,
            compute_error(a, b, points)
        )
    )
    return a,b
    
def compute_error(a, b, points):
    error = 0
    N = len(points)
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (a * x + b)) ** 2
    return error / N

def gradient_descent(starting_a, starting_b, points, learning_rate, num_iterations):
    a = starting_a
    b = starting_b
    for i in range(num_iterations):
        a, b = gradient_step(a, b, points, learning_rate)
        return a,b

def gradient_step(current_a, current_b, points, learning_rate):
    a = current_a
    b = current_b
    a_gradient = 0
    b_gradient = 0
    
    N = len(points)
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        a_gradient += - (2 / N) * x * (y - (a * x + b))
        b_gradient += - (2 / N) * (y - (a * x + b))
    
    a = current_a - learning_rate * a_gradient
    b = current_b - learning_rate * b_gradient
    return a,b

a, b = linear_regression()



