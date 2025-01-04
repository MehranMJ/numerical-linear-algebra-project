import numpy as np

def invert_upper_triangular(matrix):
    """
    Inverts an upper triangular matrix using Nath Datta's Algorithm No. 2-2-4.

    Parameters:
        matrix (np.ndarray): A square upper triangular matrix.

    Returns:
        np.ndarray: The inverted upper triangular matrix.

    Raises:
        ValueError: If the input matrix is not square or upper triangular, or if it's singular.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    n, m = matrix.shape

    if n != m:
        raise ValueError("Matrix must be square.")

    # Check if the matrix is upper triangular
    if not np.allclose(matrix, np.triu(matrix)):
        raise ValueError("Matrix must be upper triangular.")

    # Check if the matrix is singular (has zero on the diagonal)
    if np.any(np.diag(matrix) == 0):
        raise ValueError("Matrix is singular and cannot be inverted.")

    # Create an identity matrix for the inverse
    inverse = np.zeros_like(matrix)

    # Nath Datta Algorithm 2-2-4 for inversion
    for i in range(n-1, -1, -1):
        inverse[i, i] = 1 / matrix[i, i]
        for j in range(i+1, n):
            sum_term = 0
            for k in range(i+1, j+1):
                sum_term += matrix[i, k] * inverse[k, j]
            inverse[i, j] = -sum_term / matrix[i, i]

    return inverse

def main():
    """Example usage of the matrix inversion function."""
    matrix = np.array([
        [2, 1, 1],
        [0, 3, 2],
        [0, 0, 4]
    ], dtype=float)

    try:
        inverse = invert_upper_triangular(matrix)
        print("Original Matrix:")
        print(matrix)
        print("\nInverse Matrix:")
        print(inverse)

        # Verify the result
        identity_check = np.dot(matrix, inverse)
        print("\nVerification (Matrix * Inverse):")
        print(identity_check)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
