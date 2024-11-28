import numpy as np
from src.linalg_interp import gauss_iter_solve


def test_gauss_iter_solve():
    # Test 1: Solve a system with a single right-hand-side vector and compare with numpy.linalg.solve
    A1 = np.array([[4, 1, 2],
                   [3, 5, 1],
                   [1, 1, 3]])
    b1 = np.array([4, 7, 3])

    # Expected solution using numpy.linalg.solve
    expected_solution_1 = np.linalg.solve(A1, b1)

    # Test both 'seidel' and 'jacobi'
    solution_1_seidel = gauss_iter_solve(A1, b1, tol=1e-8, alg='seidel')
    solution_1_jacobi = gauss_iter_solve(A1, b1, tol=1e-8, alg='jacobi')

    tol = 1e-8
    seidel_1_pass = "Seidel passed(RHS vector)"
    seidel_1_fail = "Seidel failed(RHS vector)"
    jacobi_1_pass = "Jacobi passed(RHS vector)"
    jacobi_1_fail = "Jacobi failed(RHS vector)"

    # Iterate through the elements for Test 1
    for i in range(len(solution_1_seidel)):
        difference_1_seidel = abs(solution_1_seidel[i] - expected_solution_1[i])
        difference_1_jacobi = abs(solution_1_jacobi[i] - expected_solution_1[i])

        # Check Seidel result
        if difference_1_seidel < tol:
            result_1_seidel = seidel_1_pass
        else:
            result_1_seidel = seidel_1_fail

        # Check Jacobi result
        if difference_1_jacobi < tol:
            result_1_jacobi = jacobi_1_pass
        else:
            result_1_jacobi = jacobi_1_fail

    # Final check after the loop for Test 1
    if result_1_seidel == seidel_1_pass and result_1_jacobi == jacobi_1_pass:
        final_1 = "Test 1 passed (RHS vector)"
    else:
        final_1 = "Test 1 failed (RHS vector)"

    print(f"{result_1_seidel},\n{result_1_jacobi},\n{final_1}\n")

    # Test 2: Set up a right-hand-side matrix such that the result x is the inverse of A
    A2 = np.array([[4, 7],
                   [2, 6]])

    b2 = np.array([[1, 0],
                   [0, 1]])

    # Expected solution: inverse of A2
    expected_solution_2 = np.linalg.inv(A2)

    # Test both 'seidel' and 'jacobi'
    solution_2_seidel = gauss_iter_solve(A2, b2, tol=1e-8, alg='seidel')
    solution_2_jacobi = gauss_iter_solve(A2, b2, tol=1e-8, alg='jacobi')

    tol = 1e-8
    seidel_2_pass = "Seidel passed(Inverse of A2)"
    seidel_2_fail = "Seidel failed(Inverse of A2)"
    jacobi_2_pass = "Jacobi passed(Inverse of A2)"
    jacobi_2_fail = "Jacobi failed(Inverse of A2)"


    for i in range(len(solution_2_seidel)):
        difference_2_seidel = abs(solution_2_seidel[i] - expected_solution_2[i])
        difference_2_jacobi = abs(solution_2_jacobi[i] - expected_solution_2[i])

        # Check Seidel result
        for i in range(len(difference_2_seidel)):
            if difference_2_seidel[i] < tol:
                result_2_seidel = seidel_2_pass
            else:
                result_2_seidel = seidel_2_fail

            # Check Jacobi result
            if difference_2_jacobi[i] < tol:
                result_2_jacobi = jacobi_2_pass
            else:
                result_2_jacobi = jacobi_2_fail


    if result_2_seidel == seidel_2_pass and result_2_jacobi == jacobi_2_pass:
        final_2 = "Test 2 passed (Inverse of A2)"
    else:
        final_2 = "Test 2 failed (Inverse of A2)"

    print(f"{result_2_seidel},\n{result_2_jacobi},\n{final_2}\n")


# Run the tests
if __name__ == "__main__":
    test_gauss_iter_solve()

def test_spline_function():
    spline_function()
    if order == 1:
        linear_spline(x)

    if order == 2:
        quadratic_spline(x)

    if order == 3:
        cubic_spline(x)


