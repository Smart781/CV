from pathlib import Path

from omegaconf import OmegaConf
import cv2
import numpy as np

points = []

flag = False

def create_control_panel(cfg, window_name):
    cv2.createTrackbar("alpha", window_name, cfg.alpha_range[0], cfg.alpha_range[1], update_color_jitter)
    cv2.createTrackbar("beta", window_name, cfg.beta_range[1] * cfg.procent, cfg.beta_range[1] * cfg.procent, update_color_jitter)
    cv2.createTrackbar("gamma", window_name, cfg.gamma_range[1] * cfg.procent, cfg.gamma_range[1] * cfg.procent, update_color_jitter)
    cv2.createTrackbar("delta", window_name, cfg.delta_range[1] * cfg.procent, cfg.delta_range[1] * cfg.procent, update_color_jitter)


def get_jitter_params(cfg, window_name):
    alpha = cv2.getTrackbarPos("alpha", window_name)
    beta = cv2.getTrackbarPos("beta", window_name) / cfg.procent
    gamma = cv2.getTrackbarPos("gamma", window_name) / cfg.procent
    delta = cv2.getTrackbarPos("delta", window_name) / cfg.procent
    return alpha, beta, gamma, delta


def update_color_jitter(val):
    pass


def mouse_callback(event, x, y, flags, param):
    global points
    global flag

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 4:
            flag = True
            points = []
        points.append([x, y])


def draw_points(cfg, image, window_name):
    global points
    global flag

    for x, y in points:
        cv2.circle(image, (x, y), cfg.radius, tuple(cfg.black), -1)

    if len(points) == 4:
        # if flag:
        #     create_control_panel()
        #     flag = False

        image = perspective_transform(image, cfg.height, cfg.width)
        a, b, g, d = get_jitter_params(cfg, window_name)
        augmented_img = color_jitter(cfg, image, alpha=a, beta=b, gamma=g, delta=d)
        cv2.imshow(cfg.augmented_name, augmented_img)


def perspective_transform(image, height, width):
    src_pts = np.float32(points)
    h = height
    w = width

    dst_img = np.zeros((h, w, 3), dtype=np.uint8)

    dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    dst_img = cv2.warpPerspective(image, H, (w, h), cv2.INTER_LINEAR)

    return dst_img

def hue_augmentation(cfg, image, alpha):
    if alpha == cfg.alpha_range[0]:
        return image

    img = image.astype(np.float32) / cfg.color_size
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = np.remainder(hsv[:, :, 0] + alpha / 2, cfg.hue_board)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(img * cfg.color_size, 0, cfg.color_size).astype(np.uint8)

def saturation_augmentation(cfg, image, beta):
    if beta == cfg.beta_range[1]:
        return image

    img = image.astype(np.float32) / cfg.color_size
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Y = np.stack([Y, Y, Y], axis=-1)
    img = beta * img + (1 - beta) * Y
    return np.clip(img * cfg.color_size, 0, cfg.color_size).astype(np.uint8)

def contrast_augmentation(cfg, image, gamma):
    if gamma == cfg.gamma_range[1]:
        return image

    img = image.astype(np.float32) / cfg.color_size
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(Y, axis=(0, 1), keepdims=True)
    img = gamma * img + (1 - gamma) * mean
    return np.clip(img * cfg.color_size, 0, cfg.color_size).astype(np.uint8)

def value_augmentation(cfg, image, delta):
    if delta == cfg.delta_range[1]:
        return image

    img = image.astype(np.float32) / cfg.color_size
    img = img * delta
    return np.clip(img * cfg.color_size, 0, cfg.color_size).astype(np.uint8)

def color_jitter(cfg, image, alpha, beta, gamma, delta):
    img = hue_augmentation(cfg, image, alpha)
    img = saturation_augmentation(cfg, img, beta)
    img = contrast_augmentation(cfg, img, gamma)
    img = value_augmentation(cfg, img, delta)

    return img


def main(cfg):
    global points

    window_name = cfg.window_name
    video = cv2.VideoCapture(0)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    create_control_panel(cfg, window_name)

    while True:
        ret, image = video.read()
        if not ret:
            break

        draw_points(cfg, image, window_name)
        cv2.imshow(window_name, image)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(cfg.delay) & 0xFF == ord(cfg.symbol):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    main(cfg)