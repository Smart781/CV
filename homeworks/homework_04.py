from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


def calc_marr_hildreth_edges(cfg, log_img, T):
    H, W = log_img.shape

    pad_image = np.zeros((H + 2, W + 2))
    pad_image[1 : H + 1, 1 : W + 1] = log_img

    edges = np.zeros_like(log_img, dtype=np.uint8)

    for y in range(1, H + 1):
        for x in range(1, W + 1):
            window = pad_image[y - 1 : y + 2, x - 1 : x + 2]

            (a, b, c, d, _, e, f, g, h) = window.flatten()

            ok = False

            if d * e < 0 and np.abs(d - e) > T:
                ok = True
            elif b * g < 0 and np.abs(b - g) > T:
                ok = True
            elif a * h < 0 and np.abs(a - h) > T:
                ok = True
            elif f * c < 0 and np.abs(f - c) > T:
                ok = True

            if ok:
                edges[y - 1, x - 1] = cfg.color_size

    return edges


def ransac(cfg, edge_image, points):
    threshold = cfg.threshold
    max_trials = cfg.max_trials
    seed = cfg.seed
    max_inliers = 0
    L = []
    best_inliers = None

    rng = np.random.RandomState(seed)

    x = points[:, 0]
    y = points[:, 1]

    length = len(x)

    for i in range(max_trials):
        idx = rng.choice(length, 2, replace=False)
        x1, x2 = x[idx]
        y1, y2 = y[idx]

        a = y2 - y1
        b = -(x2 - x1)
        c = x2 * y1 - x1 * y2

        distances = np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

        x_inliers = x[distances < threshold]
        y_inliers = y[distances < threshold]

        if len(x_inliers) > max_inliers:
            max_inliers = len(x_inliers)
            best_inliers = np.column_stack((x_inliers, y_inliers))
            L = [a, b, c]

    return L, best_inliers


def compute_line_si(cfg, edge_image):
    y, x = np.where(edge_image > 0)

    points = np.column_stack((x, y))

    L, points = ransac(cfg, edge_image, points)

    L, best_inliers = ransac(cfg, edge_image, points)

    return L, best_inliers


def marr_hildreth_ransac_results(cfg):
    output_edges = cfg.output_edges
    output_image = cfg.output_image

    f = cv2.imread(cfg.input_image, cv2.IMREAD_GRAYSCALE)

    sigma = cfg.sigma
    n = 2 * np.ceil(3 * sigma).astype(int) + 1
    u = cv2.getGaussianKernel(n, sigma=0)
    G = u @ u.T

    g = cv2.filter2D(f.astype(float), ddepth=-1, kernel=G)

    L = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    log_img = cv2.filter2D(g, ddepth=-1, kernel=L)

    thresh_percent = cfg.thresh_percent
    threshold = thresh_percent * log_img.max()

    marhil_edges = calc_marr_hildreth_edges(cfg, log_img, threshold)

    line, best_inliers = compute_line_si(cfg, marhil_edges)

    dst_image = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
    a, b, c = line
    h, w = f.shape

    if b == 0:
        x = -c / a
        xy1 = (int(x), 0)
        xy2 = (int(x), h - 1)
    else:
        x1, x2 = 0, w - 1
        y1 = (-a * x1 - c) / b
        y2 = (-a * x2 - c) / b
        xy1 = (int(x1), int(y1))
        xy2 = (int(x2), int(y2))

    cv2.line(dst_image, xy1, xy2, tuple(cfg.color), cfg.line_thickness, cv2.LINE_AA)
    cv2.imwrite(output_image, dst_image)

    dst_edges = cv2.cvtColor(marhil_edges, cv2.COLOR_GRAY2RGB)

    x_inliers = best_inliers[:, 0]
    y_inliers = best_inliers[:, 1]

    for x, y in zip(x_inliers, y_inliers):
        cv2.circle(dst_edges, (int(x), int(y)), cfg.radius, tuple(cfg.color), -1)

    cv2.imwrite(output_edges, dst_edges)


def main(cfg):
    marr_hildreth_ransac_results(cfg)


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    main(cfg)
