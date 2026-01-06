#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from vision.part1_harris_corner import compute_image_gradients
from torch import nn
from typing import Tuple


"""
Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""


def get_magnitudes_and_orientations(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
    """

    # magnitudes: sqrt(Ix^2 + Iy^2)
    magnitudes = np.hypot(Ix, Iy).astype(np.float32)

    # orientations: atan2(Iy, Ix) gives angle in range [-pi, pi]
    orientations = np.arctan2(Iy, Ix).astype(np.float32)

    return magnitudes, orientations


def get_gradient_histogram_vec_from_patch(
    window_magnitudes: np.ndarray,
    window_orientations: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively. 
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram. We've already
    defined the bin centers for you, which as you can see from the np.histogram
    documentation, is passed in as the `bins` parameter and defined as a sequence
    of left bin edges.

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """
    
    NUM_BINS = 8
    bins = np.linspace(-np.pi, np.pi, NUM_BINS + 1) - 1e-5

    window_magnitudes = np.asarray(window_magnitudes)
    window_orientations = np.asarray(window_orientations)

    H, W = window_magnitudes.shape
    cell_h = H // 4
    cell_w = W // 4

    hist_list = []
    for row in range(4):
        for col in range(4):
            r0 = row * cell_h
            r1 = r0 + cell_h
            c0 = col * cell_w
            c1 = c0 + cell_w

            mags = window_magnitudes[r0:r1, c0:c1].flatten()
            orients = window_orientations[r0:r1, c0:c1].flatten()

            # use np.histogram with bins defined above; weights are magnitudes
            hist, _ = np.histogram(orients, bins=bins, weights=mags)
            hist_list.append(hist)

    wgh = np.concatenate(hist_list).astype(np.float32)
    return wgh.reshape(-1, 1)


def get_feat_vec(
    c: float,
    r: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
    
    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.
    
    Args:
        c: a float, the column (x-coordinate) of the interest point
        r: A float, the row (y-coordinate) of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    # center coordinates are (c,r) where c is x (column) and r is y (row)
    # convert to int indices; choose top-left option for even widths as in utils
    half = feature_width // 2

    # compute start/end indices (x: columns, y: rows)
    h_start = int(c) - half + 1
    h_end = int(c) + half + 1
    v_start = int(r) - half + 1
    v_end = int(r) + half + 1

    # extract patch from magnitudes and orientations
    patch_mags = magnitudes[v_start:v_end, h_start:h_end]
    patch_orients = orientations[v_start:v_end, h_start:h_end]

    # If the patch is not the expected size (near border), pad with zeros
    if patch_mags.shape != (feature_width, feature_width):
        pm = np.zeros((feature_width, feature_width), dtype=np.float32)
        po = np.zeros((feature_width, feature_width), dtype=np.float32)

        h0 = max(0, - (h_start))
        v0 = max(0, - (v_start))

        hh = min(patch_mags.shape[1], feature_width - h0)
        vv = min(patch_mags.shape[0], feature_width - v0)

        pm[v0:v0+vv, h0:h0+hh] = patch_mags[:vv, :hh]
        po[v0:v0+vv, h0:h0+hh] = patch_orients[:vv, :hh]
        patch_mags = pm
        patch_orients = po

    # compute 128-d histogram vector
    hist_vec = get_gradient_histogram_vec_from_patch(patch_mags, patch_orients)

    # L2 normalize
    norm = np.linalg.norm(hist_vec)
    if norm > 0:
        hist_vec = hist_vec / norm

    # apply square-root SIFT (element-wise sqrt)
    hist_vec = np.sqrt(hist_vec)

    fv = hist_vec.astype(np.float32)
    return fv.reshape(-1, 1)


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    # gradients
    Ix, Iy = compute_image_gradients(image_bw)

    # magnitudes and orientations
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)

    k = X.shape[0]
    feat_dim = (feature_width // 4) * (feature_width // 4) * 8
    fvs = np.zeros((k, feat_dim), dtype=np.float32)

    for i, (x, y) in enumerate(zip(X, Y)):
        fv = get_feat_vec(x, y, magnitudes, orientations, feature_width=feature_width)
        fvs[i, :] = fv.ravel()

    return fvs