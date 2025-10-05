import time
from pathlib import Path

import cv2
import numpy as np
from numba import jit, prange
from omegaconf import OmegaConf


@jit(nopython=True, parallel=True, fastmath=True)
def bilateral_filter(src_image, sigma_space, sigma_color):
    src_image = src_image.astype(np.float64)

    h, w = src_image.shape
    dst_image = np.zeros_like(src_image, dtype=np.float64)

    kernel_radius = int(3 * sigma_space)
    weights = np.zeros((2 * kernel_radius + 1, 2 * kernel_radius + 1))

    for i in range(-kernel_radius, kernel_radius + 1):
        for j in range(-kernel_radius, kernel_radius + 1):
            weights[i + kernel_radius, j + kernel_radius] = np.exp(
                -(i**2 + j**2) / (2 * sigma_space**2)
            )

    for y in prange(h):
        for x in prange(w):
            center_pixel = src_image[y, x]
            weight = 0.0
            w_sum = 0.0

            for i in range(-kernel_radius, kernel_radius + 1):
                for j in range(-kernel_radius, kernel_radius + 1):
                    if 0 <= y + i < h and 0 <= x + j < w:
                        pixel = src_image[y + i, x + j]
                        color_diff = center_pixel - pixel
                        color_weight = np.exp(-(color_diff**2) / (2 * sigma_color**2))

                        w_sum += (
                            weights[i + kernel_radius, j + kernel_radius]
                            * color_weight
                            * pixel
                        )
                        weight += (
                            weights[i + kernel_radius, j + kernel_radius] * color_weight
                        )

            if weight > 0:
                dst_image[y, x] = w_sum / weight
            else:
                dst_image[y, x] = src_image[y, x]

    dst_image = np.clip(dst_image, 0, 255).astype(np.uint8)

    assert src_image.shape == dst_image.shape
    return dst_image


def test_bilateral_filter(cfg):
    input_path = cfg.input_image_path
    output_path = cfg.output_image_path
    opencv_output_path = cfg.opencv_output_image_path
    sigma_space = cfg.sigma_space
    sigma_color = cfg.sigma_color

    src_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    print("Реализация")
    start_time = time.time()
    r_result = bilateral_filter(src_image, sigma_space, sigma_color)
    r_time = time.time() - start_time
    print("Время выполнения:", r_time, "секунд")

    print("OpenCV реализация")
    start_time = time.time()
    cv_result = cv2.bilateralFilter(
        src_image, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )
    cv_time = time.time() - start_time
    print("Время выполнения:", cv_time, "секунд")

    difference = cv2.absdiff(r_result, cv_result)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)

    print("Максимальная разница:", max_diff)
    print("Средняя разница:", mean_diff)

    cv2.imwrite(output_path, r_result)
    cv2.imwrite(opencv_output_path, cv_result)

    print("Результаты сохранены")
    print("Реализация:", output_path)
    print("OpenCV:", opencv_output_path)


def main(cfg):
    test_bilateral_filter(cfg)


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    main(cfg)
