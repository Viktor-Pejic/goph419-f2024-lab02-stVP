import numpy as np

def gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel', max_iter=1000):
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
