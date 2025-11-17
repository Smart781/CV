import numpy as np


def transform_points(points1, R, t):
    points2 = R.T @ (points1 - t).T
    return points2.T


def rotation_matrix_from_rotvec(omega):
    theta = np.linalg.norm(omega)
    n = omega / theta

    n_c = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

    idt = np.eye(3)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    R = idt + sin_th * n_c + (1 - cos_th) * (n_c @ n_c)

    return R


def rotvec_from_rotation_matrix(R):
    theta = np.arccos((np.trace(R) - 1) / 2)

    n = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    n /= 2 * np.sin(theta)
    omega = theta * n

    return omega


def project_points(points3D, P):
    n = points3D.shape[0]
    points3D_h = np.column_stack([points3D, np.ones(n)])
    points2D_h = (P @ points3D_h.T).T
    points2D = points2D_h[:, :2] / points2D_h[:, 2:3]

    return points2D


def from_image_coordinates_to_world(x, y, c, P):
    P_p = P.T @ np.linalg.inv(P @ P.T)

    x_w = np.array([x, y, 1])
    w_r_h = P_p @ x_w
    w_r = w_r_h[:3] / w_r_h[3]

    d = w_r - c
    t = -c[1] / d[1]

    point3D = c + t * d

    return point3D
