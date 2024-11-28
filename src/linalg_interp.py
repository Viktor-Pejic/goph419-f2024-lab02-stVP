import numpy as np

def gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel', max_iter=100):
    """
    Solve a linear system Ax = b using iterative methods (Gauss-Seidel or Jacobi).

    Parameters:
        A (array_like): Coefficient matrix (n x n).
        b (array_like): Right-hand-side vector(s) (n,) or (n x m).
        x0 (array_like, optional): Initial guess for the solution (n,) or (n x m). Default is None.
        tol (float, optional): Convergence tolerance. Default is 1e-8.
        alg (str, optional): Algorithm to use ('seidel' or 'jacobi'). Default is 'seidel'.
        max_iter (int, optional): Maximum number of iterations. Default is 1000.

    Returns:
        numpy.ndarray: Solution array with the same shape as b.

    Raises:
        ValueError: If input dimensions are incompatible or invalid algorithm name.
        RuntimeWarning: If the solution does not converge within max_iter iterations.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Check dimensions
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("Dimension mismatch between A and b.")

    # Initialize x0 if None
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x0 = np.array(x0, dtype=float)
        if x0.shape != b.shape:
            raise ValueError("Initial guess x0 must have the same shape as b.")
        x = x0.copy()

    alg = alg.strip().lower()
    if alg not in ('seidel', 'jacobi'):
        raise ValueError("Algorithm must be 'seidel' or 'jacobi'.")

    n = A.shape[0]

    for iteration in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            if alg == 'seidel':
                sum1 = np.dot(A[i, :i], x[:i])
                sum2 = np.dot(A[i, i+1:], x[i+1:])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]
            elif alg == 'jacobi':
                sum1 = np.dot(A[i, :i], x_old[:i])
                sum2 = np.dot(A[i, i+1:], x_old[i+1:])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Check for convergence
        if np.max(np.abs(x - x_old)) / np.max(np.abs(x)) < tol:
            return x

    # If max_iter reached without convergence
    raise RuntimeWarning(f"Solution did not converge within {max_iter} iterations.")


import numpy as np
import matplotlib.pyplot as plt

def spline_function(xd, yd, order=3):
    '''
    Generates a spline function given two vectors x and y of data.
    xd: An array_like of float data increasing in value
    yd: An array_like of float data with the same shape as xd
    order: (optional) int with possible values 1, 2, or 3 (default=3)

    ------
    return
    ------
    A function that takes one parameter (a float or array_like of float) and returns the interpolated y value(s)

    ------
    Raises
    ------
    ValueError: If input dimensions are incompatible or invalid order value.
                There are repeated values in xd.
                The xd values are not in increasing order.
                Order is a value other than 1, 2, or 3.
    '''

    # Ensure xd and yd are numpy arrays
    xd = np.array(xd, dtype=float)
    yd = np.array(yd, dtype=float)

    # Check that the input vectors have the same length
    rows_x = len(xd)
    rows_y = len(yd)

    if rows_x != rows_y:
        raise ValueError("Dimension mismatch between xd and yd.")

    # Check that there are no repeated x values
    k = len(np.unique(xd))
    if k != rows_x:
        raise ValueError("Repeating values in xd.")

    # Ensure xd is strictly increasing
    if np.any(np.diff(xd) <= 0):
        raise ValueError("xd values must be strictly increasing.")

    # Check that the order is valid (1, 2, or 3)
    if order not in (1, 2, 3):
        raise ValueError("Order must be 1, 2, or 3.")

    n = rows_x

    if order == 1:
        # Linear spline implementation
        def linear_spline(x):
            x = np.array(x, dtype=float)
            if np.any(x < xd[0]) or np.any(x > xd[-1]):
                raise ValueError(f"Input value(s) out of bounds: [{xd[0]}, {xd[-1]}].")

            # Locate intervals for x
            i = np.searchsorted(xd, x) - 1
            i = np.clip(i, 0, n - 2)  # Ensure indices are within range

            dx = x - xd[i]
            slope = (yd[i + 1] - yd[i]) / (xd[i + 1] - xd[i])  # Linear slope

            y = yd[i] + slope * dx
            return y

        return linear_spline



if __name__ == "__main__":
    xd = [0, 1, 2, 3, 4]
    yd = [0, 1, 0, -1, 0]

    # Create the spline function (linear spline in this case)
    spline_fn = spline_function(xd, yd, order=1)

    # Evaluate the spline at several points
    x = np.linspace(0, 4, 100)
    y = spline_fn(x)  # This will evaluate the spline at the given x_test points

    # Plot the results
    plt.scatter(xd, yd, color='red', label='Data Points')
    plt.plot(x, y, label='Linear Spline', color='blue')
    plt.legend()
    plt.show()





