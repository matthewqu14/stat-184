import numpy as np


def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    n, = x.shape
    es = np.identity(n)
    grad = np.array([f(x + es[i] * delta) - f(x - es[i] * delta) for i in range(n)]) / (2 * delta)

    return grad


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    m, = f(x).shape
    j = np.array([gradient(lambda v: f(v)[i], x, delta) for i in range(m)])
    
    return j


def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    def grad_func(v):
        return gradient(f, v, delta)

    return jacobian(grad_func, x, delta).T
    


