from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

points = []

flag = True


def create_control_panel(cfg, spectr_name):
    cv2.createTrackbar(
        "h",
        spectr_name,
        cfg.height_range[0] * cfg.procent,
        cfg.height_range[1] * cfg.procent,
        update_spectr,
    )
    cv2.createTrackbar(
        "w",
        spectr_name,
        cfg.width_range[0] * cfg.procent,
        cfg.width_range[1] * cfg.procent,
        update_spectr,
    )


def get_params(cfg, spectr_name):
    h = cv2.getTrackbarPos("h", spectr_name) / cfg.procent
    w = cv2.getTrackbarPos("w", spectr_name) / cfg.procent

    return h, w


def update_spectr(val):
    global flag

    flag = True


def draw_spectr(cfg, image, spectr_name, window_name):
    global flag

    f = np.float32(image)

    H, W = f.shape
    cx, cy = W // 2, H // 2
    h, w = get_params(cfg, spectr_name)

    F = np.fft.fftshift(np.fft.fft2(f))

    abs_F = np.abs(F)
    arg_F = np.angle(F)

    abs_F[0 : int(cy * h) + 1, int(cx * (1 - w)) : int(cx * (1 + w)) + 1] = 0
    abs_F[
        (-int(cy * h) if h != 0 else H) : H + 1,
        int(cx * (1 - w)) : int(cx * (1 + w)) + 1,
    ] = 0

    spectrum = np.log(1 + abs_F)

    spectrum_image = np.uint8(
        cfg.color_size * (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    )

    cv2.imshow(spectr_name, spectrum_image)

    F = abs_F * np.exp(1j * arg_F)

    g = np.fft.ifft2(np.fft.ifftshift(F))
    g = np.real(g)

    g_image = np.uint8(cfg.color_size * (g - g.min()) / (g.max() - g.min()))

    cv2.imshow(window_name, g_image)

    flag = False


def main(cfg):
    global flag

    window_name = cfg.window_name
    spectr_name = cfg.spectr_name

    image = cv2.imread(cfg.image, cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow(window_name)
    cv2.namedWindow(spectr_name)

    create_control_panel(cfg, spectr_name)

    while True:
        if flag:
            draw_spectr(cfg, image, spectr_name, window_name)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(cfg.delay) & 0xFF == ord(cfg.symbol):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    main(cfg)
