import cv2
import numpy as np


def compute_homography_dlt(points1, points2):
    N = len(points1)
    Omega = np.zeros((2 * N, 9))

    for i in range(N):
        x, y = points1[i]
        x1, y1 = points2[i]

        Omega[2 * i] = [0, 0, 0, -x, -y, -1, y1 * x, y1 * y, y1]
        Omega[2 * i + 1] = [x, y, 1, 0, 0, 0, -x1 * x, -x1 * y, -x1]

    U, S, V = np.linalg.svd(Omega)
    theta = V[-1, :]
    H = theta.reshape(3, 3)
    H /= H[2, 2]

    return H


def stabilize_video_with_homography(frames):
    N = frames.shape[0]
    stabilized_frames = np.zeros_like(frames)
    image1 = frames[0].copy()
    stabilized_frames[0] = image1
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    points1 = cv2.goodFeaturesToTrack(gray1, 2000, 0.01, 10)
    points1 = points1[:, 0, :]

    for i in range(1, N):
        curr_image = frames[i]
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY)
        points_curr, _, _ = cv2.calcOpticalFlowPyrLK(gray1, curr_gray, points1, None)

        H, _ = cv2.findHomography(points_curr, points1, cv2.RANSAC)
        height, width = image1.shape[:2]
        warped_image = cv2.warpPerspective(curr_image, H, (width, height))
        stabilized_frames[i] = warped_image

    assert frames.shape == stabilized_frames.shape

    return stabilized_frames


def get_affine_transform(points1, points2, image1):
    N = len(points1)

    A = np.zeros((2 * N, 6))
    b = np.zeros((2 * N, 1))

    for i in range(N):
        x, y = points1[i]
        x1, y1 = points2[i]

        A[2 * i] = [x, y, 1, 0, 0, 0]
        b[2 * i] = x1

        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i + 1] = y1

    theta, res, r, s = np.linalg.lstsq(A, b, rcond=None)
    A = np.array(
        [
            [theta[0, 0], theta[1, 0], theta[2, 0]],
            [theta[3, 0], theta[4, 0], theta[5, 0]],
        ]
    )

    points1_h = np.hstack([points1, np.ones((N, 1))])
    points_pred = (A @ points1_h.T).T
    errors = np.linalg.norm(points2 - points_pred, axis=1)
    proj_error = np.mean(errors)

    return A, proj_error
