import numpy as np
from scipy.spatial.transform import Rotation


def solvePnP_DLT(points3D, points2D, K):
    size = len(points3D)
    K_inv = np.linalg.inv(K)
    points2D_h = np.hstack([points2D, np.ones((size, 1))])
    points2D_norm = (K_inv @ points2D_h.T).T

    A = []

    for i in range(size):
        X, Y, Z = points3D[i]
        u, v, w = points2D_norm[i]

        eq1 = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        eq2 = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

        A.append(eq1)
        A.append(eq2)

    A = np.array(A)

    U, S, V = np.linalg.svd(A)
    theta = V[-1, :]

    P = theta.reshape(3, 4)

    R = P[:, :3]
    t = P[:, 3]

    U_r, S_r, V_r = np.linalg.svd(R)
    R_ort = U_r @ V_r

    scale = np.mean(S_r)
    t /= scale

    if np.linalg.det(R_ort) < 0:
        R_ort = -R_ort
        t = -t

    rot = Rotation.from_matrix(R_ort)
    omega = rot.as_rotvec()

    return omega, t


def triangulate_DLT(rotation_vectors, translations, camera_points2D, K):
    cam_size = len(rotation_vectors)
    p_size = len(camera_points2D[0])
    K_inv = np.linalg.inv(K)
    points3D_r = []

    for point_ind in range(p_size):
        A = []
        b = []

        for cam_ind in range(cam_size):
            omega = rotation_vectors[cam_ind]
            x_img, y_img = camera_points2D[cam_ind][point_ind]

            point_img_h = np.array([x_img, y_img, 1.0])
            point_norm = K_inv @ point_img_h
            x_pr, y_pr, w_pr = point_norm

            R = Rotation.from_rotvec(omega).as_matrix()

            r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
            r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
            r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
            t1, t2, t3 = translations[cam_ind]

            a1 = [r31 * x_pr - r11, r32 * x_pr - r12, r33 * x_pr - r13]
            b1 = t1 - t3 * x_pr

            a2 = [r31 * y_pr - r21, r32 * y_pr - r22, r33 * y_pr - r23]
            b2 = t2 - t3 * y_pr

            A.append(a1)
            A.append(a2)
            b.append(b1)
            b.append(b2)

        A = np.array(A)
        b = np.array(b)

        w, res, r, s = np.linalg.lstsq(A, b, rcond=None)
        points3D_r.append(w)

        points3D = np.array(points3D_r)

    return points3D
