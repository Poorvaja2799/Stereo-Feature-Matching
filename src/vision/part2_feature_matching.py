#!/usr/bin/python3

import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed. We recommend using one for-loop 
    at the appropriate level if you run out of memory.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    dists = np.zeros((features1.shape[0], features2.shape[0]))
    for i,feat in enumerate(features1):
        dists[i,:] = np.linalg.norm(features2 - feat[np.newaxis, :], axis=1)

    return dists


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    # handle empty inputs
    if features1.size == 0 or features2.size == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    # compute pairwise distances (n1 x n2)
    dists = compute_feature_distances(features1, features2)

    # for each feature in features1, find indices of nearest and second nearest
    nn_idx = np.argsort(dists, axis=1)

    matches = []
    confidences = []
    ratio_thresh = 0.8

    for i in range(dists.shape[0]):
        if dists.shape[1] < 2:
            # cannot apply ratio test if there's only one candidate
            continue

        j1 = int(nn_idx[i, 0])
        j2 = int(nn_idx[i, 1])
        d1 = dists[i, j1]
        d2 = dists[i, j2]

        # avoid division by zero
        if d2 <= 0:
            continue

        ratio = d1 / d2
        if ratio < ratio_thresh:
            matches.append((i, j1))
            confidences.append(1.0 - ratio)

    if len(matches) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    matches = np.array(matches, dtype=int)
    confidences = np.array(confidences, dtype=float)

    # sort by confidence descending
    order = np.argsort(confidences)[::-1]
    matches = matches[order]
    confidences = confidences[order]

    return matches, confidences
