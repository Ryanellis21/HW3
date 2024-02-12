import math

def is_symmetric(matrix):
    """
    Check if a matrix is symmetric.

    Args:
        matrix (list): The matrix to check.

    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.

    Args:
        matrix (list): The matrix to check.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    try:
        cholesky_decomposition(matrix)
        return True
    except Exception:
        return False

def cholesky_decomposition(matrix):
    """
    Perform the Cholesky decomposition of a matrix.

    Args:
        matrix (list): The matrix to decompose.

    Returns:
        list: The lower triangular matrix L.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = (matrix[i][i] - s) ** 0.5
            else:
                L[i][j] = (1.0 / L[j][j] * (matrix[i][j] - s))
    return L

def forward_substitution(L, b):
    """
    Perform forward substitution to solve a lower triangular system.

    Args:
        L (list): The lower triangular matrix.
        b (list): The right-hand side vector.

    Returns:
        list: The solution vector y.
    """
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def backward_substitution(L, y):
    """
    Perform backward substitution to solve an upper triangular system.

    Args:
        L (list): The upper triangular matrix.
        y (list): The right-hand side vector.

    Returns:
        list: The solution vector x.
    """
    n = len(L)
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(L[i][j] * x[j] for j in range(i+1, n))) / L[i][i]
    return x

def cholesky_solve(matrix, b):
    """
    Solve a linear system using Cholesky decomposition.

    Args:
        matrix (list): The coefficient matrix.
        b (list): The right-hand side vector.

    Returns:
        list: The solution vector x.
    """
    L = cholesky_decomposition(matrix)
    y = forward_substitution(L, b)
    x = backward_substitution(L, y)
    return x

def doolittle_solve(matrix, b):
    """
    Solve a linear system using Doolittle's LU decomposition.

    Args:
        matrix (list): The coefficient matrix.
        b (list): The right-hand side vector.

    Returns:
        list: The solution vector x.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = matrix[i][j] - s

        for j in range(i+1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (matrix[j][i] - s) / U[i][i]

    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def solve_matrix_equation(matrix, b):
    """
    Solve a matrix equation Ax = b using Cholesky or Doolittle method.

    Args:
        matrix (list): The coefficient matrix A.
        b (list): The right-hand side vector b.

    Returns:
        None
    """
    if is_symmetric(matrix) and is_positive_definite(matrix):
        print("Using Cholesky method:")
        x = cholesky_solve(matrix, b)
    else:
        print("Using Doolittle method:")
        x = doolittle_solve(matrix, b)
    print("Solution vector x:")
    print(x)

# Problem 1
A1 = [[1, -1, 3, 2],
      [-1, 5, -5, -2],
      [3, -5, 19, 3],
      [2, -2, 3, 21]]
b1 = [15, -35, 94, 1]

# Problem 2
A2 = [[4, 2, 4, 0],
      [2, 2, 3, 2],
      [4, 3, 6, 3],
      [0, 2, 3, 9]]
b2 = [20, 36, 60, 122]

print("Problem 1:")
solve_matrix_equation(A1, b1)
print("\nProblem 2:")
solve_matrix_equation(A2, b2)