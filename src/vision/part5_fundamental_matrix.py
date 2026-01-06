"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N,2) array")

    # Compute centroid
    centroid = np.mean(pts, axis=0)

    # Translate points so centroid is at origin
    pts_centered = pts - centroid

    # Compute per-axis standard deviation and use 1/std as scale so each
    # axis has unit variance after normalization (0-mean, unit variance)
    stds = np.std(pts_centered, axis=0)
    # Avoid division by zero
    scale_x = 1.0 / stds[0] if stds[0] > 0 else 1.0
    scale_y = 1.0 / stds[1] if stds[1] > 0 else 1.0

    # Construct normalization matrix T: scale * translate
    T = np.array(
        [
            [scale_x, 0.0, -scale_x * centroid[0]],
            [0.0, scale_y, -scale_y * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    # Apply transformation in homogeneous coords
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    pts_norm_h = (T @ pts_h.T).T

    points_normalized = pts_norm_h[:, :2]

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """

    F_norm = np.asarray(F_norm)
    T_a = np.asarray(T_a)
    T_b = np.asarray(T_b)

    if F_norm.shape != (3, 3):
        raise ValueError("F_norm must be 3x3")
    if T_a.shape != (3, 3) or T_b.shape != (3, 3):
        raise ValueError("T_a and T_b must be 3x3 transformation matrices")

    F_orig = T_b.T @ F_norm @ T_a

    return F_orig


def make_singular(F_norm: np.array) -> np.ndarray:
    """
    Force F to be singular by zeroing the smallest of its singular values.
    This is done because F is not supposed to be full rank, but an inaccurate
    solution may end up as rank 3.

    Args:
    - F_norm: A numpy array of shape (3,3) representing the normalized fundamental matrix.

    Returns:
    - F_norm_s: A numpy array of shape (3, 3) representing the normalized fundamental matrix
                with only rank 2.
    """
    U, D, Vt = np.linalg.svd(F_norm)
    D[-1] = 0
    F_norm_s = np.dot(np.dot(U, np.diag(D)), Vt)

    return F_norm_s


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here. Equation (9) in the documentation indicates
    one equation of a linear system in which you'll want to solve for f_{i, j}.

    Since the matrix is defined up to a scale, many solutions exist. To constrain
    your solution, use can either use SVD and use the last Vt vector as your
    solution, or you can fix f_{3, 3} to be 1 and solve with least squares.

    Be sure to reduce the rank of your estimate - it should be rank 2. The
    make_singular() function can do this for you.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    pts_a = np.asarray(points_a)
    pts_b = np.asarray(points_b)
    if pts_a.shape != pts_b.shape:
        raise ValueError("points_a and points_b must have the same shape")
    if pts_a.ndim != 2 or pts_a.shape[1] != 2:
        raise ValueError("points must be of shape (N,2)")

    N = pts_a.shape[0]
    if N < 8:
        raise ValueError("At least 8 point correspondences are required to estimate the fundamental matrix")

    # Normalize points
    a_norm, T_a = normalize_points(pts_a)
    b_norm, T_b = normalize_points(pts_b)

    # Build design matrix A (N x 9). Equation: x'^T F x = 0
    A = np.zeros((N, 9))
    for i in range(N):
        x, y = a_norm[i]
        x_p, y_p = b_norm[i]
        A[i] = [x_p * x, x_p * y, x_p, y_p * x, y_p * y, y_p, x, y, 1.0]

    # Solve Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]  # last row (smallest singular value)
    F_norm = f.reshape(3, 3)

    # Enforce rank-2 constraint
    F_norm = make_singular(F_norm)

    # Unnormalize
    F = unnormalize_F(F_norm, T_a, T_b)

    return F
