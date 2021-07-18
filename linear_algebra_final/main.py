import numpy as np
import fractions


def create_matrix():
    print("2x2 matrix:\n[ m11  m12 ]")
    print("[ m21  m22 ]")
    m11 = float(input("Enter the value of m11: "))
    m12 = float(input("Enter the value of m12: "))
    m21 = float(input("Enter the value of m21: "))
    m22 = float(input("Enter the value of m22: "))
    mat = np.matrix([[m11,m12], [m21, m22]])
    mat_array = np.array([[m11,m12], [m21, m22]])
    print(mat)
    print("\nExponential of Matrix: ")
    print(np.dot(np.array([[m11,m12],[m21,m22]]), np.array([[m11,m12],[m21,m22]])))
    return mat, mat.trace(), mat_array


def transpose_matrix(mat):
    mat_trs = np.matrix.getT(mat)
    print(mat_trs)
    return mat_trs


def invert_matrix(mat):
    mat_inv = np.matrix.getI(mat)
    print(mat_inv)
    return mat_inv


def determinant_matrix(mat):
    mat_det = np.linalg.det(mat)
    print(str(round(mat_det, 4)))
    return mat_det


def rref(A, tol=1.0e-12):
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A, jb


def eigen_matrix(mat):
    eigval_mat, eigvec_mat = np.linalg.eig(mat)
    print(eigval_mat)
    print("\nEigenvectors of Matrix: ")
    print(eigvec_mat)


# The main function is the first thing the program will run when started.
if __name__ == '__main__':
    # Makes it so the output is printed as fractions instead of decimals.
    np.set_printoptions(formatter={'all': lambda x: str(fractions.Fraction(x).limit_denominator())})
    print("Matrix Equation Solver Thingy!\nby Cameron MacDonald | chm7374\n")
    mat, mat_trace, mat_array = create_matrix()
    print("\nTranspose of matrix: ")
    mat_trs = transpose_matrix(mat)
    print("\nInverse of matrix: ")
    mat_inv = invert_matrix(mat)
    print("\nDeterminant of matrix: ", end="")
    mat_det = determinant_matrix(mat)
    print("\nReduced row matrix: ")
    rref_mat, j = rref(mat)
    print(rref_mat)
    print("\nRank of Matrix:", np.linalg.matrix_rank(mat))
    str_trace = str(mat_trace)[2:]
    str_trace = str_trace[:1]
    print("Trace of Matrix:", str_trace)
    np.set_printoptions(precision=5)
    print("\nEigenvalues of Matrix: ")
    eigen_matrix(mat_array)
