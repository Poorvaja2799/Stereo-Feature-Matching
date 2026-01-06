import math

import numpy as np
import cv2


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: int
) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    # Probability that a single randomly drawn sample of size `sample_size`
    # is all inliers
    p_individual = float(ind_prob_correct) ** sample_size

    # If the sample success probability is 1, only one iteration is needed
    if p_individual >= 1.0:
        return 1

    # If the sample success probability is 0, we can never succeed
    if p_individual <= 0.0:
        return math.inf

    # Use the standard formula: 1 - (1 - p_individual)^S >= prob_success
    # => (1 - p_individual)^S <= 1 - prob_success
    # => S >= log(1 - prob_success) / log(1 - p_individual)
    # Guard against log domain errors
    if prob_success <= 0.0:
        return 0
    if prob_success >= 1.0:
        # require infinite samples in theory; but return large number
        return math.inf

    denom = math.log(1.0 - p_individual)
    numer = math.log(1.0 - prob_success)

    # Compute required number of iterations
    S = numer / denom

    # Ceil to integer number of iterations
    num_samples = math.ceil(S)

    return int(num_samples)

def ransac_homography(
    points_a: np.ndarray, points_b: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Uses the RANSAC algorithm to robustly estimate a homography matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) of points from image A.
    -   points_b: A numpy array of shape (N, 2) of corresponding points from image B.

    Returns:
    -   best_H: The best homography matrix of shape (3, 3).
    -   inliers_a: The subset of points_a that are inliers (M, 2).
    -   inliers_b: The subset of points_b that are inliers (M, 2).
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    #                                                                         #
    # HINT: You are allowed to use the `cv2.findHomography` function to       #
    # compute the homography from a sample of points. To compute a direct     #
    # solution without OpenCV's built-in RANSAC, use it like this:            #
    #   H, _ = cv2.findHomography(sample_a, sample_b, 0)                      #
    # The `0` flag ensures it computes a direct least-squares solution.       #
    ###########################################################################

    pts_a = np.asarray(points_a)
    pts_b = np.asarray(points_b)

    if pts_a.ndim != 2 or pts_b.ndim != 2 or pts_a.shape[1] != 2 or pts_b.shape[1] != 2:
        raise ValueError("points must be Nx2 arrays")
    if pts_a.shape[0] != pts_b.shape[0]:
        raise ValueError("points_a and points_b must have the same number of points")

    # Ensure OpenCV gets float32 input (required on some platforms)
    pts_a_cv = pts_a.astype(np.float32)
    pts_b_cv = pts_b.astype(np.float32)

    # Use OpenCV RANSAC to estimate homography
    ransac_thresh = 3.0
    H, mask = cv2.findHomography(pts_a_cv, pts_b_cv, cv2.RANSAC)

    if H is None:
        # fallback: compute direct homography from all points
        H, _ = cv2.findHomography(pts_a_cv, pts_b_cv, 0)
        mask = np.ones((pts_a.shape[0], 1), dtype=np.uint8)

    # mask can be Nx1 with values 0/1 or 0/255; convert to boolean
    if mask is None:
        mask_bool = np.ones((pts_a.shape[0],), dtype=bool)
    else:
        mask_arr = np.asarray(mask).reshape(-1)
        mask_bool = mask_arr != 0

    inliers_a = pts_a[mask_bool]
    inliers_b = pts_b[mask_bool]

    # Normalize H so bottom-right is 1 (if possible)
    try:
        if np.abs(H[2, 2]) > 1e-8:
            H = H / H[2, 2]
    except Exception:
        pass

    best_H = H

    return best_H, inliers_a, inliers_b